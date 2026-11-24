"""平台模型回执诊断与规则降级，不记录完整响应或异常消息。"""
from src.harness.round_controller import apply_verdict, extract_text, parse_verdict


def decide_with_receipt(rnd, prompt, budget, history, round_index):
    diagnostics = {"status": "call_error", "error_type": "", "finish_reason": "",
                   "response_keys": [], "text_chars": 0}
    text = ""
    verdict = None
    try:
        receipt = rnd.model([{"role": "user", "content": prompt}],
                            max_tokens=min(256, rnd.max_output_tokens), temperature=0)
        response = receipt.response
        if isinstance(response, dict):
            diagnostics["response_keys"] = sorted(str(k) for k in response)[:20]
            choices = response.get("choices")
            first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            reason = first.get("finish_reason")
            diagnostics["finish_reason"] = reason[:80] if isinstance(reason, str) else ""
        text = extract_text(response)
        verdict = parse_verdict(text)
        diagnostics.update(text_chars=len(text),
                           status="valid_json" if verdict else "invalid_json" if text else "empty")
    except Exception as exc:  # 通道失败回退规则；保留真实调用ID，仍尝试 finish
        diagnostics["error_type"] = type(exc).__name__
    decision = apply_verdict(budget, verdict, history, round_index)
    keys = ("action", "epochs", "lr")
    changed = verdict is not None and any(decision.get(k) != budget.get(k) for k in keys)
    decision["source"] = "llm" if changed else "rule"
    diagnostics["changed_rule"] = changed
    return decision, text, diagnostics
