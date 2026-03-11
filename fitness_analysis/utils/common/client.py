import os
import json
import re
from django.http import Http404
from ...models import Recording

class OpenAIClient:
    """封裝 OpenAI API 相關功能"""

    def __init__(self):
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"

    def get_response(self, prompt: str) -> str:
        """一次性取得完整回應"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content.strip()

    def stream_response(self, prompt: str):
        """Generator：串流回傳回應 chunk"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield {"data": delta.content}
            yield {"event": "end", "data": ""}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    @staticmethod
    def extract_markdown(llm_output: str) -> str:
        """移除 LLM 回應中的 markdown code fence"""
        return re.sub(r'^```(?:markdown)?\n([\s\S]+?)\n```$', r'\1', llm_output.strip())

    @staticmethod
    def _get_sport_name_zh(sport: str) -> str:
        if not sport:
            return "健身"
        sport_map = {
            'deadlift': '硬舉',
            'benchpress': '臥推'
        }
        return sport_map.get(sport.lower(), sport.title())

    def get_suggestion(self, recording_id: int) -> dict:
        """根據 recording_id 讀取 Score.json，產生健身建議"""
        from django.conf import settings
        try:
            recording = Recording.objects.get(id=recording_id)
        except Recording.DoesNotExist:
            raise Http404("Recording not found")

        # 將相對路徑轉為絕對路徑
        folder = os.path.join(settings.BASE_DIR, recording.folder)
        score_json_path = os.path.join(folder, "config/Score.json")
        print(f"[OpenAIClient.get_feedback] Score.json path: {score_json_path}")

        if not os.path.exists(score_json_path):
            raise FileNotFoundError(f"Score.json not found: {score_json_path}")

        with open(score_json_path, mode='r', encoding='utf-8') as f:
            score_data = json.load(f)['results']

        sport_name_zh = self._get_sport_name_zh(recording.sport)
        prompt = (
            f"你是一個健身教練，這是我在做一組{sport_name_zh}訓練時發生的錯誤--{score_data}，"
            "key為第幾組，value包含了錯誤動作的信心值以及這組的總分，"
            "請根據這些資訊判斷(回答以信心值作為嚴重程度的標準，例如:可能、有一點、明顯等等，不要直接出現信心值)，"
            "越大表示該錯誤錯得越明顯，請問要怎麼修正他的動作，指出哪一下有問題。"
            "請用 markdown 語法回答，給我英文回答"
        )
        result = self.get_response(prompt)
        return {'result': self.extract_markdown(result)}

    def get_suggestion_stream(self, recording_id: int):
        """根據 recording_id 讀取 Score.json，產生『串流格式』的健身建議"""
        from django.conf import settings
        try:
            recording = Recording.objects.get(id=recording_id)
        except Recording.DoesNotExist:
            yield f"data: {json.dumps({'error': 'Recording not found'})}\n\n"
            return

        folder = os.path.join(settings.BASE_DIR, recording.folder)
        score_json_path = os.path.join(folder, "config/Score.json")

        if not os.path.exists(score_json_path):
            yield f"data: {json.dumps({'error': 'Score.json not found'})}\n\n"
            return

        with open(score_json_path, mode='r', encoding='utf-8') as f:
            score_data = json.load(f)['results']

        sport_name_zh = self._get_sport_name_zh(recording.sport)
        prompt = (
            f"你是一個健身教練，這是我在做一組{sport_name_zh}訓練時發生的錯誤--{score_data}，"
            "key為第幾組，value包含了錯誤動作的信心值以及這組的總分，"
            "請根據這些資訊判斷(回答以信心值作為嚴重程度的標準，例如:可能、有一點、明顯等等，不要直接出現信心值)，"
            "越大表示該錯誤錯得越明顯，請問要怎麼修正他的動作，指出哪一下有問題。"
            "請用 markdown 語法回答，給我英文回答"
        )
        
        # 使用串流模式
        for chunk in self.stream_response(prompt):
            # 目前 stream_response 回傳的是 {"data": "..."} 或 {"event": "...", "data": "..."}
            yield f"data: {json.dumps(chunk)}\n\n"
            
    def get_workout_plan(self, recording_id: int) -> dict:
        """根據 recording_id 讀取 Score.json，產生健身菜單"""
        from django.conf import settings
        try:
            recording = Recording.objects.get(id=recording_id)
        except Recording.DoesNotExist:
            raise Http404("Recording not found")

        # 將相對路徑轉為絕對路徑
        folder = os.path.join(settings.BASE_DIR, recording.folder)
        score_json_path = os.path.join(folder, "config/Score.json")
        print(f"[OpenAIClient.get_feedback] Score.json path: {score_json_path}")

        if not os.path.exists(score_json_path):
            raise FileNotFoundError(f"Score.json not found: {score_json_path}")

        with open(score_json_path, mode='r', encoding='utf-8') as f:
            score_data = json.load(f)['results']
        sport_name_zh = self._get_sport_name_zh(recording.sport)
        prompt = (
            f"你是一個健身教練，這是我在做一組{sport_name_zh}訓練時發生的錯誤--{score_data}，"
            "key 為第幾組，value 包含了錯誤動作分類模型的信心值"
            "(回答以信心值作為嚴重程度的標準，例如：可能、有一點、明顯等等，不要直接出現信心值)。"
            "你會建議他怎麼「安排一週詳細的訓練菜單」，越清楚越好。"
            "請用 markdown 語法回答，給我英文回答"
        )
        
        result = self.get_response(prompt)
        return {'result': self.extract_markdown(result)}

    def get_workout_plan_stream(self, recording_id: int):
        """根據 recording_id 讀取 Score.json，產生『串流格式』的健身建議"""
        from django.conf import settings
        try:
            recording = Recording.objects.get(id=recording_id)
        except Recording.DoesNotExist:
            yield f"data: {json.dumps({'error': 'Recording not found'})}\n\n"
            return

        folder = os.path.join(settings.BASE_DIR, recording.folder)
        score_json_path = os.path.join(folder, "config/Score.json")

        if not os.path.exists(score_json_path):
            yield f"data: {json.dumps({'error': 'Score.json not found'})}\n\n"
            return

        with open(score_json_path, mode='r', encoding='utf-8') as f:
            score_data = json.load(f)['results']

        sport_name_zh = self._get_sport_name_zh(recording.sport)
        prompt = (
            f"你是一個健身教練，這是我在做一組{sport_name_zh}訓練時發生的錯誤--{score_data}，"
            "key 為第幾組，value 包含了錯誤動作分類模型的信心值"
            "(回答以信心值作為嚴重程度的標準，例如：可能、有一點、明顯等等，不要直接出現信心值)。"
            "你會建議他怎麼「安排一週詳細的訓練菜單」，越清楚越好。"
            "請用 markdown 語法回答，給我英文回答"
        )
        
        # 使用串流模式
        for chunk in self.stream_response(prompt):
            # 目前 stream_response 回傳的是 {"data": "..."} 或 {"event": "...", "data": "..."}
            yield f"data: {json.dumps(chunk)}\n\n"
