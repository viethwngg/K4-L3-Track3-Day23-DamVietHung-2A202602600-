# Survey of LLM agents and tool use
## TL;DR
- LLM agents emerged from early browser- and modular-reasoning systems into explicit reasoning-plus-action loops, with ReAct as one of the clearest early articulations of interleaved thought and action and Toolformer/Gorilla showing how tool selection and API calling can be trained or specialized rather than only prompted [1][2][3][4][5].
- Modern agent architectures increasingly separate planning from execution, add explicit reflection or reviewer roles, and externalize state into planners, executors, and memory/context modules to sustain long-horizon tasks [6][7][8][9].
- Evaluation has moved beyond final-answer accuracy toward benchmark families for multi-step tool calls, multimodal step-level reasoning, learning from interaction, and red-teaming for deception, prompt injection, and harmful multi-step behavior [10][11][12][13][14][15].
- The newest 2026 evidence suggests the frontier is shifting from isolated tool calls to stateful simulations, embodied and physical environments, and long-horizon business or game workflows, while safety work now treats deception and compositional harm as first-class agent risks [16][8][13][14][17][15].
- Across the retrieved sources, the main limitation is not a lack of agent activity but a lack of standardized cross-benchmark comparability, because many papers expose only abstract-level summaries, project READMEs, or narrow domain metrics [6][10][12][16][17][15].

## Background
LLM agents are systems that use a language model not only to generate text but also to choose and execute actions in an external environment, such as web browsing, API calls, software tools, or embodied control. In the retrieved literature, the foundational move is to connect language generation with external action: WebGPT uses a text-based browsing environment and human feedback, MRKL proposes a modular neuro-symbolic architecture with discrete reasoning and knowledge modules, and ReAct explicitly interleaves reasoning traces with task-specific actions [1][2][3].

A second foundational thread is tool-specific learning. Toolformer teaches models when to call APIs, how to format calls, and how to incorporate returned results into prediction, while Gorilla shows how an API-connected model can be paired with retrieval to track changing documentation and write calls for large API suites [4][5]. HuggingGPT extends the idea from single tools to orchestration: an LLM planner can decompose a task, select specialist models by function description, execute subtasks, and summarize the outcome [18].

## Foundational agent loops and tool-use interfaces
The most important conceptual shift in early agent work is from passive text generation to an action loop. WebGPT demonstrates browsing as an environment for question answering, but it remains bounded to one domain; ReAct generalizes the idea by defining a loop where reasoning traces help induce, track, and update actions across tasks like knowledge retrieval, fact verification, embodied navigation, and web shopping [1][3]. That makes ReAct the clearest retrieved foundation for the modern “think, act, observe, repeat” agent pattern [3].

MRKL contributes a complementary systems view: instead of a single monolithic model, a flexible architecture can route between neural components and discrete modules for knowledge and reasoning [2]. In practice, this perspective matters because many later systems treat a model as a planner or controller rather than as the sole executor of all subproblems [2][18].

Toolformer and Gorilla show two distinct tool-use interfaces. Toolformer learns tool invocation decisions in a self-supervised way over several tool types, emphasizing the model’s ability to decide when to call and how to incorporate results [4]. Gorilla instead focuses on API-call correctness at scale and uses document retrieval to adapt to changing APIs, which is closer to production tool calling [5]. Together they suggest that tool use can be learned either as a generic capability or as an API-specific competency [4][5].

## Prompting, planning, and control architectures
Recent agent systems increasingly distribute control across multiple roles. PC-Agent decomposes decisions into instruction, subtask, and action levels, and assigns manager, progress, decision, and reflection roles so that planning, tracking, execution, and feedback are separated [6]. Recursive Game Creator uses a similarly layered loop with designer, builder, player, and reviewer roles, where the reviewer’s feedback recursively informs the next design round [8]. These systems are evidence that multi-role coordination has become a standard answer to long-horizon complexity [6][8].

Planner-executor separation is another recurring design. BrowseMaster explicitly splits search strategy formulation from retrieval execution so that one component preserves long-horizon reasoning while another performs targeted retrieval [19]. SuperNav applies the same logic in embodied navigation: the multimodal model interprets requests and makes decisions, while navigation tools execute motion, letting the model stay general-purpose instead of being overfit to navigation action prediction [9].

Reflection is now used in at least two different ways. PC-Agent uses a Reflection agent for bottom-up error feedback and adjustment, while CLEAR uses agentic reflection upstream as a context-generation mechanism that turns prior trajectories into task-specific summaries for later decision making [6][7]. That distinction matters: one version of reflection corrects ongoing execution, while the other creates better context before execution begins [6][7].

Finally, training and optimization methods increasingly shape tool-use behavior directly. APIGen curates verified function-calling data through format checking, function execution, and semantic verification, showing that supervised data quality is now part of agent design [19]. RC-GRPO pushes further by treating multi-turn tool calling as a reward-conditioned exploration problem, which suggests that policy optimization for agents now explicitly targets variance, diversity, and multi-turn credit assignment rather than just next-token accuracy [20].

## Evaluation, benchmarks, and failure modes
Benchmarking of LLM agents has broadened rapidly. MCPToolBench++ focuses on MCP tool use, including single-step and multi-step calls, and highlights practical evaluation pain points such as heterogeneous response formats, varying success rates across servers, and context limits caused by long tool descriptions [12]. Agent-X evaluates deep multimodal reasoning with step-level scoring, showing that multi-step vision tasks remain hard even for strong models [10]. Learn2Play Bench measures whether agents actually learn from repeated interaction in unfamiliar environments rather than merely exploiting pretrained knowledge [15].

Safety and robustness benchmarks now examine adversarial behavior, not just competence. AgentVigil targets indirect prompt injection, demonstrating that agents can be misled into unsafe navigation actions and that attacks transfer across settings [21]. AgentHazard studies harmful behavior that emerges from individually plausible steps composing into dangerous multi-step trajectories, which reframes safety as a long-horizon property rather than a single bad action [11]. DecepEval adds deception as a distinct axis and treats inducement, pressure, opportunity, and conflict as conditions that can raise deceptive behavior [13].

The recurring failure modes across these benchmarks are structurally similar. Tool-use agents struggle with context overload and tool-description overload in MCP settings, weak full-chain success in multimodal tasks, poor adaptation from experience in unfamiliar environments, and security vulnerabilities under indirect prompt injection [12][10][15][21]. On the safety side, the evidence shows that autonomous agents may deceive when incentives are misaligned and may produce harmful outcomes through compositionally unsafe action sequences [13][11].

## Recent developments and where the field is going
The 2024-10-09 to 2026-10-09 window shows a clear move toward stateful and long-horizon agent evaluation. Business Arena evaluates agents as end-to-end business operators over a long market horizon, so the agent must manage persistent obligations, noisy evidence, and delayed returns rather than isolated tasks [17]. Learn2Play Bench similarly asks whether agents can improve across repeated exposure to unfamiliar games, making experience and adaptation central to evaluation [15].

A second recent direction is agentic simulation. Trace2Env proposes training-free reconstruction of textual environments from traces so that agents can be trained and tested in safe, reproducible sandboxes even when the original system is unavailable [16]. This is important because it turns historical interaction traces into an evaluation substrate, which may lower the cost of testing long-horizon behavior [16].

A third direction is embodied and physical-world generalization. RobotWorld extends tool-using agent evaluation to robot use across manipulation, locomotion, driving, and aerial control, while SuperNav tries to preserve general-purpose reasoning and delegate motion to tools rather than fine-tuning the language model into a narrow controller [14][9]. The broader implication is that tool use is no longer only about software APIs; it now spans simulated physics and physical interfaces [14][9].

## Trends and open problems
The main trend is architectural externalization: recent systems increasingly split planning, execution, reflection, and memory across separate modules rather than asking one model to do everything [6][7][8][9]. This improves controllability and long-horizon handling, but the retrieved abstracts do not yet show a consensus on the best division of labor or the best prompt/action schema [6][19][8][9].

A second trend is evaluation expansion. Benchmarks are now probing long horizons, interaction learning, deception, harmful composition, prompt injection, and embodied operation, which is a sign that tool use is becoming an end-to-end systems problem rather than a narrow function-calling problem [10][11][12][13][14][17][15]. The open problem is that these benchmarks use incompatible environments, metrics, and success notions, so cross-paper comparisons remain weak [10][12][17][15].

A third trend is that safety concerns are becoming more agent-specific. The new evidence emphasizes deception, injection, and multi-step harm, which are all properties that only emerge over sequences of decisions and observations [21][11][13]. What is still missing is a standard safety protocol that jointly measures competence, honesty, and resistance to malicious tool/environment feedback [21][11][13].

Finally, the strongest open methodological gap is reproducibility at the level of agent traces and tool schemas. Many of the retrieved pages are abstract summaries or repository READMEs, so they establish the direction of the field but not full experimental detail [16][8][14][17][15][9]. For a mature survey field, the next step is likely a shared trace format, shared tool protocol, and shared long-horizon evaluation suite that can compare architectures, training methods, and safety defenses on a common scale [12][16][17][9].

## References
[1] WebGPT: Browser-assisted question-answering with human feedback. web. https://arxiv.org/abs/2112.09332 (2021-12-17)
[2] MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning modules. web. https://arxiv.org/abs/2205.00445 (2022-05-01)
[3] ReAct: Synergizing Reasoning and Acting in Language Models. web. https://arxiv.org/abs/2210.03629 (2022-10-06)
[4] Toolformer: Language Models Can Teach Themselves to Use Tools. web. https://arxiv.org/abs/2302.04761 (2023-02-09)
[5] Gorilla: Large Language Model Connected with Massive APIs. web. https://arxiv.org/abs/2305.15334 (2023-05-24)
[6] PC-Agent: A Hierarchical Multi-Agent Collaboration Framework for Complex Task Automation on PC. hf-search. https://huggingface.co/papers/2502.14282 (2025-02-20)
[7] CLEAR: Context Augmentation from Contrastive Learning of Experience via Agentic Reflection. hf-search. https://huggingface.co/papers/2604.07487 (2026-04-08)
[8] Recursive Game Creator: An Agentic Product-Level Experience-Oriented Game Harness. hf-daily. https://huggingface.co/papers/2610.08621 (2026-10-06)
[9] SuperNav: An Agentic Navigation System for Any Task in Any Scene. hf-daily. https://huggingface.co/papers/2610.12126 (2026-10-08)
[10] Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks. web. https://huggingface.co/papers/2505.24876 (2025-05-30)
[11] AgentHazard: A Benchmark for Evaluating Harmful Behavior in Computer-Use Agents. web. https://huggingface.co/papers/2604.02947 (2026-04-03)
[12] MCPToolBench++: A Large Scale AI Agent Model Context Protocol MCP Tool Use Benchmark. web. https://huggingface.co/papers/2508.07575 (2025-08-11)
[13] DecepEval: A Benchmark for Evaluating Deception in LLM Agents. hf-daily. https://huggingface.co/papers/2610.07967 (2026-10-06)
[14] RobotWorld: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments. hf-daily. https://huggingface.co/papers/2610.10409 (2026-10-07)
[15] Learn2Play Bench: How Well Do LLM Agents Learn from Experience in Unfamiliar Environments?. web. https://huggingface.co/papers/2610.08215 (2026-10-08)
[16] From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation. hf-daily. https://huggingface.co/papers/2610.06100 (2026-10-05)
[17] Business Arena: A long-horizon benchmark for agents that run an end-to-end business. hf-search. https://huggingface.co/papers/2608.08621 (2026-08-09)
[18] HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face. web. https://arxiv.org/abs/2303.17580 (2023-03-30)
[19] APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets. hf-search. https://huggingface.co/papers/2406.18518 (2024-06-26)
[20] RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents. hf-search. https://huggingface.co/papers/2602.03025 (2026-02-03)
[21] AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt Injection against LLM Agents. web. https://huggingface.co/papers/2505.05849 (2025-05-09)
