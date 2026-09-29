"""GLM request routes validate context before streaming and render an empty think block without a reasoning-effort line when thinking is off."""

from __future__ import annotations

import os
from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest, RequestError


# OpenAI reasoning_effort -> GLM-5.3's template levels (it knows low and high; anything else renders as max)
EFFORTS = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}


def with_effort(body: dict[str, Any], default: str | None = None) -> dict[str, Any]:
    """The request with its ``reasoning_effort`` as template kwargs: "none" turns thinking off, a level picks the
    template's Reasoning Effort line; explicit chat_template_kwargs win. Without one, ``default``
    (``TF_GLM_DEFAULT_EFFORT``) applies, and with neither the template thinks at max."""

    effort = body.get("reasoning_effort")
    kwargs0 = body.get("chat_template_kwargs") or {}
    if not isinstance(effort, str) and default and "reasoning_effort" not in kwargs0 and \
            kwargs0.get("enable_thinking", True) is not False:
        effort = default
    if not isinstance(effort, str):
        return body
    effort = effort.strip().lower()
    kwargs = dict(body.get("chat_template_kwargs") or {})
    if effort == "none":
        kwargs.setdefault("enable_thinking", False)
    elif effort in EFFORTS:
        kwargs.setdefault("reasoning_effort", EFFORTS[effort])
    else:
        return body
    return {**body, "chat_template_kwargs": kwargs}


class ThinkingOffTemplate:
    """The checkpoint's chat template, rendered as GLM-5.3's thinking-off template renders it when thinking is off."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def render(self, messages, *, tools, enable_thinking, extra=None, allow_images=False) -> str:
        text = self.inner.render(messages, tools=tools, enable_thinking=enable_thinking, extra=extra,
                                 allow_images=allow_images)
        if not enable_thinking:
            text = text.replace("<|system|>Reasoning Effort: Max", "", 1)
            if text.endswith("<|assistant|><think>"):
                text += "</think>"
        return text


class GlmApp(App):
    reads_ignore_eos = True             # ``run`` hands it to the engine's request

    def __init__(self, engine, model_dir, served: str, **kwargs: Any) -> None:
        super().__init__(engine, model_dir, served, **kwargs)
        self.template = ThinkingOffTemplate(self.template)
        if getattr(engine, "tower", None) is not None and not getattr(engine, "concurrent", False):
            from .vision import Frontend        # images: each placeholder becomes its keyed run

            self.vision = Frontend(self.tok, engine.w.cfg.image_token)

    def _prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        return super()._prepare(with_effort(body, os.environ.get("TF_GLM_DEFAULT_EFFORT") or None), chat)

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Validate the rendered prompt plus max_tokens against the engine context limit before streaming."""

        problem = self._check_fields(body)
        limit = getattr(self.engine, "limit", None)
        if problem or limit is None:
            return problem or super().check(body, prepared=prepared)
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        prompt = len(prepared.prompt)
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        need = prompt + (int(asked) if asked else 1)
        if need <= limit:
            return super().check(body, prepared=prepared)
        detail = f"{prompt} prompt tokens plus max_tokens {int(asked)}" if asked else f"a {prompt}-token prompt"
        return (f"this request needs a {need}-token context ({detail}), and this server was started for {limit}: "
                f"shorten the prompt or reply{self._restart(need, ' both ranks')}")

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        model = str(body.get("model") or "")
        self.engine.request.policy = body.get("tf_policy") or (model.split("@", 1)[1] if "@" in model else None)
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        return super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)
