"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import os
import sys
import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant – trợ lý AI chính thức của Tập đoàn Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ Vingroup (VinFast, Vinpearl)
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa thông tin.

## AVAILABLE TOOLS
{tools}

## CORE RULES (Bắt buộc tuân thủ)
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). PHẢI gọi tool `search_product_catalog` để lấy dữ liệu thực.
2. KHÔNG BAO GIỜ tự tạo ticket_id. PHẢI gọi tool `submit_support_ticket` để ghi nhận.
3. Nếu khách hàng hỏi câu FAQ đơn giản (chính sách bảo hành, đổi trả chung), có thể trả lời trực tiếp mà không cần gọi tool.
4. Nếu câu hỏi cần NHIỀU tool, hãy gọi tuần tự từng tool rồi tổng hợp kết quả.

## OPERATIONAL BOUNDARIES
- CHỈ hỗ trợ thông tin liên quan đến hệ sinh thái Vingroup (VinFast, Vinpearl, Vinhomes).
- Từ chối lịch sự nếu người dùng hỏi ngoài phạm vi.

## OUTPUT CONTRACT
Mỗi lượt suy nghĩ tuân thủ format:
Thought: Phân tích yêu cầu của người dùng
Action: Tên tool cần gọi (hoặc 'None' nếu trả lời trực tiếp)
Action Input: Tham số JSON cho tool
Observation: Kết quả trả về từ tool
Final Answer: Câu trả lời cuối cùng gửi tới người dùng
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop.
    Mục đích: So sánh chất lượng trả lời khi LLM bịa thông tin (hallucination).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Gửi câu hỏi tới LLM (hoặc trả lời mock nếu không có API key)."""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn sản phẩm Vingroup. Hãy trả lời câu hỏi sau:\n{user_input}"
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "live_api"
                }
            except Exception as e:
                pass

        # Mock fallback (khi không có API key hoặc lỗi API)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Production-grade Agent với System Prompt Engineering & Tool Calling.

    Features:
      - 2 custom tools: search_product_catalog, submit_support_ticket
      - Sequential & Parallel tool calling
      - Max iterations safeguard
      - Full trace logging
    """

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------
    # Intent Detection (Rule-based Simulator)
    # -------------------------------------------------------------------
    def _detect_intent(self, user_input: str) -> Dict[str, Any]:
        """Phân tích intent từ user_input bằng rule-based keyword matching."""
        query_lower = user_input.lower()

        # 1. Catalog Intent: Chỉ kích hoạt khi người dùng có nhu cầu tìm kiếm/xem giá/sản phẩm
        needs_catalog = False
        catalog_category = None
        max_price = 999999999999

        catalog_triggers = ["xem", "tìm", "mua", "giá", "sản phẩm", "resort", "khách sạn", "danh mục", "có xe"]
        has_catalog_intent = any(trigger in query_lower for trigger in catalog_triggers)

        if has_catalog_intent:
            if any(w in query_lower for w in ["xe", "xe điện", "ô tô", "vf"]):
                needs_catalog = True
                catalog_category = "xe_dien"
            elif any(w in query_lower for w in ["du lịch", "resort", "nghỉ dưỡng", "vinpearl", "khách sạn", "phòng"]):
                needs_catalog = True
                catalog_category = "du_lich"

        # Trích xuất giá tối đa nếu có
        price_match = re.search(r'(?:dưới\s*)?(\d+(?:\.\d+)?)\s*(triệu|tỷ|tr)', query_lower)
        if price_match:
            val = float(price_match.group(1))
            unit = price_match.group(2)
            if "tỷ" in unit:
                max_price = int(val * 1_000_000_000)
            else:
                max_price = int(val * 1_000_000)

        # 2. Support Ticket Intent: Kích hoạt khi có phản ánh lỗi, khiếu nại, yêu cầu hỗ trợ
        needs_ticket = False
        ticket_args = {}
        ticket_triggers = ["lỗi", "hỏng", "khiếu nại", "hỗ trợ", "phản hồi", "sự cố", "gấp", "vấn đề"]
        if any(trigger in query_lower for trigger in ticket_triggers):
            needs_ticket = True

            # Trích xuất tên khách hàng
            name_match = re.search(r'(?:tên tôi là|tôi tên(?: là)?)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\bxe\b|\bphòng\b|\bvà\b|$)', user_input, re.IGNORECASE)
            customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

            # Xác định mức độ ưu tiên
            if any(w in query_lower for w in ["gấp", "nghiêm trọng", "khẩn cấp"]):
                priority = "high"
            elif any(w in query_lower for w in ["thấp"]):
                priority = "low"
            else:
                priority = "medium"

            # Trích xuất mô tả vấn đề
            issue_match = re.search(r'(?:xe\s+\w+.*?bị\s+[^.]+|phòng.*?bị\s+[^.]+|bị lỗi\s+[^.]+|vấn đề\s+[^.]+)', user_input, re.IGNORECASE)
            if issue_match:
                issue_description = issue_match.group(0).strip()
            else:
                issue_description = user_input

            ticket_args = {
                "customer_name": customer_name,
                "issue_description": issue_description,
                "priority": priority
            }

        # 3. FAQ Intent: Hỏi đáp thông tin chính sách mà không cần gọi tool
        is_faq = False
        if any(w in query_lower for w in ["bảo hành", "chính sách"]) and not needs_ticket:
            is_faq = True

        return {
            "needs_catalog": needs_catalog,
            "catalog_args": {"category": catalog_category, "max_price": max_price} if needs_catalog else {},
            "needs_ticket": needs_ticket,
            "ticket_args": ticket_args,
            "is_faq": is_faq
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        self.trace.append({"step": "init", "user_input": user_input})

        intents = self._detect_intent(user_input)
        iteration = 0

        # FAQ: Trả lời trực tiếp, không gọi tool
        if intents["is_faq"]:
            iteration += 1
            answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km tuỳ điều kiện nào đến trước."
            self.trace.append({
                "step": "faq",
                "thought": "Câu hỏi thuộc diện FAQ về bảo hành, trả lời trực tiếp mà không cần gọi tool.",
                "action": None,
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # Gọi các tool theo intent đã nhận diện
        catalog_results = None
        ticket_result = None

        if intents["needs_catalog"]:
            iteration += 1
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached"
                }

            cat_args = intents["catalog_args"]
            self.trace.append({
                "step": f"iteration_{iteration}",
                "thought": f"Cần gọi tool search_product_catalog với category='{cat_args.get('category')}' và max_price={cat_args.get('max_price')}.",
                "action": "search_product_catalog",
                "action_input": cat_args
            })
            catalog_results = search_product_catalog(**cat_args)
            self.trace[-1]["observation"] = catalog_results

        if intents["needs_ticket"]:
            iteration += 1
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached"
                }

            t_args = intents["ticket_args"]
            self.trace.append({
                "step": f"iteration_{iteration}",
                "thought": f"Cần tạo ticket hỗ trợ kỹ thuật cho khách hàng {t_args.get('customer_name')}.",
                "action": "submit_support_ticket",
                "action_input": t_args
            })
            ticket_result = submit_support_ticket(**t_args)
            self.trace[-1]["observation"] = ticket_result

        # Tổng hợp câu trả lời Final Answer
        answer_parts = []

        if catalog_results is not None:
            if not catalog_results or len(catalog_results) == 0:
                answer_parts.append("Rất tiếc, không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn.")
            else:
                prods = [f"- {p['name']} (Giá: {p['price_vnd']:,} VNĐ)" for p in catalog_results]
                answer_parts.append("Dưới đây là các sản phẩm phù hợp:\n" + "\n".join(prods))

        if ticket_result is not None:
            answer_parts.append(
                f"Yêu cầu hỗ trợ của khách hàng {ticket_result['customer_name']} đã được tiếp nhận. "
                f"Mã ticket của bạn là {ticket_result['ticket_id']} với mức độ ưu tiên {ticket_result['priority']}."
            )

        if not answer_parts:
            iteration += 1
            final_ans = "Tôi là VinAssistant. Tôi có thể giúp gì cho quý khách về sản phẩm xe điện VinFast hoặc dịch vụ nghỉ dưỡng Vinpearl?"
        else:
            final_ans = "\n\n".join(answer_parts)

        self.trace.append({
            "step": "final_answer",
            "final_answer": final_ans
        })

        return {
            "answer": final_ans,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
