from datetime import datetime
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class MathSolverAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = "math_solver_user"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def _build_agent(self) -> LlmAgent:
        return LlmAgent(
            model="gemini-1.5-flash-latest",
            name="maths_solver_agent",
            description="Solves mathematical expressions and returns the result",
            instruction="Reply with the calculated output."
        )

    def invoke(self, query: str, session_id: str) -> str:
        session = self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id
        )

        if session is None:
            session = self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                session_id=session_id,
                state={}
            )

        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)]
        )

        events = list(self._runner.run(
            user_id=self._user_id,
            session_id=session.id,
            new_message=content
        ))

        if not events or not events[-1].content or not events[-1].content.parts:
            return ""

        return "\n".join([p.text for p in events[-1].content.parts if p.text])

    async def stream(self, query: str, session_id: str):
        # This is a stub method for streaming, it needs implementation
        result = await self.invoke(query, session_id)
        yield {
            "is_task_complete": True,
            "content": f"The result is: {result}"
        }
