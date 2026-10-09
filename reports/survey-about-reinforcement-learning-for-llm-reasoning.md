# Survey about reinforcement learning for LLM reasoning

## TL;DR
- RL for LLM reasoning evolved from early RLHF-style preference optimization into verifier- and reward-aware training recipes that more directly target reasoning quality, with recent surveys explicitly unifying RLHF and RLVR under one policy-gradient view. [1][2]
- The strongest direct reasoning gains in the retrieved evidence come from verifier-driven methods, including step-aware verification, process-verifier scaling, and formal-theorem-proving RL with verifier feedback, rather than from early alignment papers that mostly report helpfulness or harmlessness gains. [3][4][5]
- Beyond standard RLHF, recent work explores process reward models, self-play or search-guided RL, rejection-sampling hybrids, and generator-verifier co-training; several papers report better sample efficiency or robustness, but the methods are often domain-specific. [6][7][8][9][10][11][12]
- In the 2024-10 to 2026-10 window, the field increasingly focuses on scaling rules, adaptive exploration, and benchmark design to test whether RL truly improves reasoning rather than merely overfitting to easier items or existing solution modes. [13][14][15][16][17]
- Evaluation work highlights brittle failure modes such as reward hacking, verifier gaming, and overfitting to uncertain reward models, so benchmark quality and protocol design are now central to claims about RL-enhanced reasoning. [18][19][20][21][22][23]

## Background
Early RLHF established the basic post-training recipe: preference data or reward modeling is used to steer a policy, and alternative formulations such as pairwise/K-wise comparisons, ranking losses, and scalable training pipelines made this approach more practical. The retrieved evidence shows that these methods were mainly validated on alignment, helpfulness, and harmlessness tasks rather than on broad reasoning benchmarks. [1][24][25][26]

A key conceptual shift is that reasoning benefits are more likely when the reward signal itself is structured around reasoning steps or verification, not just end-of-answer preference. The step-aware verifier paper and the later survey of RL post-training both support this view by linking RLHF, RLVR, and process-oriented reward design as related but distinct training families. [3][2]

## Foundations: RLHF, RLAIF, and scalable post-training
The foundational family in the retrieved sources is standard RLHF and its close variants. The pairwise/K-wise comparison framework provides a theoretical basis for preference-based optimization, RRHF replaces PPO-style optimization with ranking loss, and DeepSpeed-Chat demonstrates that RLHF-style pipelines can be made scalable and affordable for large models. These sources establish the infrastructure and objective-shaping ideas that later reasoning-focused methods inherit. [1][24][25]

RLAIF extends this foundation by using AI-generated preferences instead of human feedback. The retrieved evidence says RLAIF can match RLHF on summarization and dialogue alignment, while direct RLAIF can be superior to canonical RLAIF, which matters because it reduces the human-label bottleneck even though the evidence here is not reasoning-specific. [26]

The limitation of this foundation is that its retrieved evidence does not directly demonstrate reasoning improvements. In this corpus, the most concrete reasoning gains come later, when reward and verification become step-aware or domain-checked rather than purely preference-based. [26][1][24][25][3]

## Verifier-shaped reward: the most direct route to reasoning gains
Verifier-shaped methods provide the clearest link between RL and reasoning quality. Step-aware verifier methods report large gains on math reasoning, and the verifier-integrated theorem-proving paper shows that Lean 4 feedback and multi-turn verifier interaction can improve MiniF2F pass@128. These are narrower domains than general chat, but they offer the strongest evidence that RL helps when correctness can be checked reliably. [3][4]

Process-verifier work strengthens this conclusion by arguing that rewards should measure progress at each step rather than only final correctness. The paper reports that process advantage verifiers can outperform outcome reward models in search efficiency and sample efficiency, suggesting that dense step-level credit assignment is a meaningful improvement for reasoning tasks. [5]

The survey of RL post-training places these approaches under RLVR and related methods, emphasizing that reward design and verification are now central to reasoning-oriented post-training. That framing helps explain why later papers increasingly use rule-based rewards, formal verifiers, or process rewards instead of generic RLHF-style preference signals. [2][5]

## Beyond standard RLHF: process rewards, self-play, search, and rejection-sampling hybrids
A second family of methods goes beyond standard RLHF by changing how credit is assigned or how trajectories are generated. The retrieved evidence includes process reward models such as VersaPRM, self-play or exploration-separated training such as RLSP, and search-oriented RL such as SSRL. Together, these methods try to reward progress, exploration, or internal search behaviors rather than only the final answer. [7][8][27]

Rejection-sampling-plus-RL hybrids are another important branch. The minimalist RAFT/Reinforce-Rej paper argues that training only on positively rewarded samples can be surprisingly competitive with GRPO and PPO, while GVM-RAFT adds dynamic sample allocation to reduce gradient variance and improve both speed and accuracy. This suggests that part of the recent progress comes from simplifying the training recipe rather than making the optimizer more complex. [9][10]

Generator-verifier co-training further blurs the line between training and evaluation. RL Tango jointly trains a generator and verifier, while the compute-optimal solving paper shows that, at many budgets, generating more candidate solutions can be more efficient than heavy verification. Taken together, these results imply that reasoning systems may need a budget-aware mix of solution generation, verification, and rejection rather than a single dominant algorithm. [11][12]

## Recent developments and benchmark pressure, 2024-10 to 2026-10
The most recent period in the retrieved corpus shifts from “can RL help?” to “under what scaling laws does RL help?” Logic-RL reports that rule-based RL with carefully designed prompts and format rewards can generalize from a small synthetic logic set to math benchmarks, while SWEET-RL extends RL to multi-turn collaborative reasoning with step-level rewards. These papers broaden the scope from single-turn math to more agentic reasoning settings. [13][14]

A parallel thread studies the RL-versus-distillation tradeoff. The retrieved comparison paper argues that RLVR can raise pass@1 without necessarily improving broader capability, while distillation improves capability only when new knowledge is added; this is an important warning that better accuracy on benchmark items does not automatically mean deeper reasoning. [15]

Benchmark design has become more adversarial. Depth-Breadth Synergy argues that RLVR gains depend on both adaptive exploration depth and training breadth, while MATH-Beyond claims many existing math benchmarks are saturated at high sampling budgets and therefore fail to test whether RL exceeds the base model’s capabilities. Learn2Play Bench and the hf-daily RLVR distillation paper both push evaluation toward unfamiliar or post-hoc-experience settings where agents must learn from interaction or from hindsight rather than memorize solutions. [16][17][28]

## Trends and open problems
A major trend is the move toward step-level, process-level, and verification-based reward design. The evidence suggests that these methods can improve sample efficiency and direct reasoning performance, but they often require domains where correctness can be checked or where verifiers are themselves strong enough to be useful. [3][5][7][8][11]

A second trend is that benchmark saturation is now a methodological problem, not just a practical inconvenience. Several recent papers argue that standard math benchmarks are too easy for high-sampling policies or too biased toward easier items, so future progress claims will need harder, more interactive, or more distribution-shifted evaluations. [15][16][17]

A third trend is that reward and verifier failure modes are increasingly treated as first-class research questions. RewardBench, the reward-design paper, the uncertainty-aware RLHF paper, the verifier-gaming paper, Hard2Verify, and the fact-verifier study all show that reward uncertainty, annotation noise, extensional shortcutting, and reward hacking can distort apparent progress. [18][19][20][21][22][23]

The open problem is how to combine accuracy, capability, and robustness in one training/evaluation loop. The retrieved evidence suggests that purely outcome-based RL can overfit or hack rewards, while richer verifier or process rewards can be expensive, narrow, or vulnerable to their own shortcuts. A credible survey conclusion is therefore that RL for LLM reasoning is promising, but only when paired with harder benchmarks, stronger verifiers, and evaluation protocols that expose shortcut behavior. [5][15][18][19][20][21][22][23]

## References
[1] Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons. web. https://arxiv.org/abs/2301.11270 (2023-01-26)
[2] Reinforcement Learning for LLM Post-Training: A Survey. web. https://arxiv.org/abs/2407.16216 (2024-07-23)
[3] Making Large Language Models Better Reasoners with Step-Aware Verifier. web. https://arxiv.org/abs/2206.02336 (2022-06-06)
[4] Leanabell-Prover-V2: Verifier-integrated Reasoning for Formal Theorem Proving via Reinforcement Learning. web. https://arxiv.org/abs/2507.08649 (2025-07-11)
[5] Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning. web. https://arxiv.org/abs/2410.08146 (2024-10-10)
[6] Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning. hf-search. https://huggingface.co/papers/2508.08221 (2025-08-11)
[7] On the Emergence of Thinking in LLMs I: Searching for the Right Intuition. web. https://arxiv.org/abs/2502.06773 (2025-02-10)
[8] VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data. web. https://arxiv.org/abs/2502.06737 (2025-02-10)
[9] A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce. web. https://arxiv.org/abs/2504.11343 (2025-04-15)
[10] Optimizing Chain-of-Thought Reasoners via Gradient Variance Minimization in Rejection Sampling and RL. web. https://arxiv.org/abs/2505.02391 (2025-05-05)
[11] RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning. web. https://arxiv.org/abs/2505.15034 (2025-05-21)
[12] When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning. web. https://arxiv.org/abs/2504.01005 (2025-04-01)
[13] Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning. web. https://arxiv.org/abs/2502.14768 (2025-02-20)
[14] SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks. web. https://arxiv.org/abs/2503.15478 (2025-03-19)
[15] Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning. web. https://arxiv.org/abs/2505.14216 (2025-05-20)
[16] Depth-Breadth Synergy in RLVR: Unlocking LLM Reasoning Gains with Adaptive Exploration. web. https://arxiv.org/abs/2508.13755 (2025-08-19)
[17] Learn2Play Bench: How Well Do LLM Agents Learn from Experience in Unfamiliar Environments?. hf-daily. https://huggingface.co/papers/2610.08215 (2026-10-08)
[18] RewardBench: Evaluating Reward Models for Language Modeling. web. https://arxiv.org/abs/2403.13787 (2024-03-20)
[19] On Designing Effective RL Reward at Training Time for LLM Reasoning. web. https://arxiv.org/abs/2410.15115 (2024-10-19)
[20] Towards Reliable Alignment: Uncertainty-aware RLHF. web. https://arxiv.org/abs/2410.23726 (2024-10-31)
[21] LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking. web. https://arxiv.org/abs/2604.15149 (2026-04-16)
[22] Hard2Verify: A Step-Level Verification Benchmark for Open-Ended Frontier Math. web. https://arxiv.org/abs/2510.13744 (2025-10-15)
[23] Verifying the Verifiers: Unveiling Pitfalls and Potentials in Fact Verifiers. web. https://arxiv.org/abs/2506.13342 (2025-06-16)
[24] RRHF: Rank Responses to Align Language Models with Human Feedback without tears. web. https://arxiv.org/abs/2304.05302 (2023-04-11)
[25] DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales. web. https://arxiv.org/abs/2308.01320 (2023-08-02)
[26] RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. hf-search. https://huggingface.co/papers/2309.00267 (2023-09-01)
[27] SSRL: Self-Search Reinforcement Learning. web. https://arxiv.org/abs/2508.10874 (2025-08-14)
[28] Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight. hf-daily. https://huggingface.co/papers/2610.08077 (2026-10-06)
