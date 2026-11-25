"""平台模型回执诊断与规则降级，不记录完整响应或异常消息。"""
import json

from src.harness.round_controller import (
    apply_verdict, extract_text_with_source, parse_truncated_verdict, parse_verdict,
)


def decide_with_receipt(rnd, prompt, budget, history, round_index):
    diagnostics = {"status": "call_error", "error_type": "", "finish_reason": "",
                   "response_keys": [], "text_source": "", "text_chars": 0, "max_tokens": None}
    summary = ""
    verdict = None
    try:
        limit = min(768, int(getattr(rnd, "max_output_tokens", 768) or 768))
        if limit <= 0:
            raise ValueError
        diagnostics["max_tokens"] = limit
        receipt = rnd.model([{"role": "user", "content": prompt}], max_tokens=limit, temperature=0)
        response = receipt.response
        if isinstance(response, dict):
            diagnostics["response_keys"] = sorted(str(k) for k in response)[:20]
            choices = response.get("choices")
            first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            reason = first.get("finish_reason")
            diagnostics["finish_reason"] = reason[:80] if isinstance(reason, str) else ""
        text, channel = extract_text_with_source(response)
        verdict = parse_verdict(text, allow_truncated=False)
        status = "valid_json" if verdict else "invalid_json" if text else "empty"
        if text and verdict is None and diagnostics["finish_reason"] == "length":
            verdict = parse_truncated_verdict(text, budget)
            if verdict is not None:
                status = "truncated_json_recovered"
        diagnostics.update(text_chars=len(text), text_source=channel, status=status)
        if verdict is not None:
            # Both return values flow into state/audit/finish. Never persist raw
            # text or free-form reasoning, even when the JSON itself is valid.
            verdict["reason"] = "truncated_json_recovered" if status == "truncated_json_recovered" else ""
            summary = json.dumps({k: verdict[k] for k in ("action", "epochs", "lr")}, separators=(",", ":"))
    except Exception as exc:  # 通道失败回退规则；保留真实调用ID，仍尝试 finish
        verdict = None
        diagnostics.update(status="call_error", error_type=type(exc).__name__)
    decision = apply_verdict(budget, verdict, history, round_index)
    keys = ("action", "epochs", "lr")
    changed = verdict is not None and any(decision.get(k) != budget.get(k) for k in keys)
    decision["source"] = "llm" if changed else "rule"
    diagnostics["changed_rule"] = changed
    return decision, summary, diagnostics
