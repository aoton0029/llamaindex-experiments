import logging
from llama_index.core.agent import (
    AgentInput, 
    AgentOutput, 
    AgentChatResponse, 
    AgentStream, 
    AgentWorkflow, 
    ReActAgent, 
    CodeActAgent, 
    FunctionAgent, 
    BaseWorkflowAgent, 
    AgentStreamStructuredOutput
)

logger = logging.getLogger(__name__)

class AgentFactory:
    @staticmethod
    def create(agent_type, **kwargs):
        if agent_type == "react":
            return ReActAgent(**kwargs)
        elif agent_type == "codeact":
            return CodeActAgent(**kwargs)
        elif agent_type == "function":
            return FunctionAgent(**kwargs)
        else:
            raise ValueError(f"未知のエージェントタイプ: {agent_type}")
        
    
    @staticmethod
    def create_workflow_agent(workflow: AgentWorkflow, **kwargs) -> BaseWorkflowAgent:
        try:
            agent = BaseWorkflowAgent(workflow=workflow, **kwargs)
            logger.info("Workflowエージェントを作成")
            return agent
        except Exception as e:
            logger.error(f"Workflowエージェント作成エラー: {e}")
            raise