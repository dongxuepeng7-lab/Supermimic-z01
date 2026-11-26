# SG-AMP: 10-minute English conference talk

12 main slides, 600 seconds in total. Two hidden backup slides support Q&A. The English script is also embedded in PowerPoint speaker notes. Timings include short pauses and figure explanation. Rehearse once with a timer and adjust pace.

## Timing

| Slide | Topic | Duration | Cumulative |
|---|---|---:|---:|
| 1 | SG-AMP | 30 s | 0:30 |
| 2 | Natural gait and stable contact | 45 s | 1:15 |
| 3 | SG-AMP: three changes to AMP | 50 s | 2:05 |
| 4 | G1: CoP-gated style reward | 65 s | 3:10 |
| 5 | G2: a phase-conditioned discriminator | 55 s | 4:05 |
| 6 | G3: stability-driven style curriculum | 50 s | 4:55 |
| 7 | Simulation protocol | 45 s | 5:40 |
| 8 | Robustness improves in both stress tests | 65 s | 6:45 |
| 9 | Lower jitter and more planar contact | 55 s | 7:40 |
| 10 | Each module contributes to performance | 65 s | 8:45 |
| 11 | The gate attenuates style more under stress | 40 s | 9:25 |
| 12 | Conclusion | 35 s | 10:00 |

## Slide 1: SG-AMP

Suggested time: 30 seconds.

Good morning, everyone. Today I will present SG-AMP, a method for learning natural and robust bipedal locomotion. This is joint work by Jiahong Li, Xuepeng Dong, Shengli Xu, and Beiyan Jiang. Our main idea is to regulate motion imitation using a physical stability signal. I will first explain the motivation, then introduce the method, and finally discuss our simulation results.

## Slide 2: Natural gait and stable contact

Suggested time: 45 seconds.

Reinforcement learning can produce robust walking, but the resulting gait may be stiff or excessively bent at the knees. Adversarial Motion Priors, or AMP, address naturalness by rewarding transitions that resemble reference motion. However, this introduces three difficulties. First, the style score does not directly describe contact stability. Second, a discriminator without gait phase can provide inconsistent feedback within a walking cycle. Third, a fixed style weight creates a difficult tradeoff when balance becomes challenging. The snapshots here show the SG-AMP gait. Our research question is how to preserve useful motion style while adapting its influence to the robot’s current stability.

## Slide 3: SG-AMP: three changes to AMP

Suggested time: 50 seconds.

SG-AMP keeps the standard AMP and PPO training structure, and changes three specific parts. G1 multiplies the style reward by a gate computed from the Center of Pressure, or CoP. G2 adds a gait phase encoding to the discriminator. G3 adapts the style weight using a moving average of the gate. In the objective at the bottom, the task reward still encourages locomotion. The AMP style reward is scaled by both the instantaneous gate and the adaptive weight. These two factors operate at different time scales. The design uses contact forces and foot geometry already available in the physics backend, and each component can be removed independently for ablation.

## Slide 4: G1: CoP-gated style reward

Suggested time: 65 seconds.

The first module connects style learning to contact geometry. We compute the CoP as a vertical-force-weighted average of the contact positions. The support polygon is constructed from the active foot geometry, and the margin is the minimum signed distance from the CoP to its boundary. We then map that margin to a continuous gate using a sigmoid. The threshold is 2.5 centimeters, and the smoothing scale is 6 millimeters. When the margin is large, the gate approaches one and preserves the style signal. Near the support boundary, it reduces the style contribution, allowing the task objective to have more influence. This is a multiplication on the scalar reward and does not add a new gradient path. We use the margin as a stability-related training signal. It should not be interpreted as a formal guarantee against falling.

## Slide 5: G2: a phase-conditioned discriminator

Suggested time: 55 seconds.

The second module addresses consistency within a gait cycle. The same joint configuration can have a different meaning during support and swing. A discriminator that ignores this context may give fluctuating style feedback. We maintain a phase clock using the commanded step frequency and align or reset it at heel-strike events. The phase enters the discriminator as cosine and sine, together with the state transition. This encoding represents a periodic variable continuously across the cycle boundary. Reference trajectories receive the same phase labels offline, so the discriminator compares motion under a consistent phase convention. The intended effect is smoother style supervision and less high-frequency joint motion. The ablation results later show how much spectral energy increases when this phase input is removed.

## Slide 6: G3: stability-driven style curriculum

Suggested time: 50 seconds.

An instantaneous gate can respond to a difficult contact, but it does not by itself determine the appropriate overall intensity of imitation. The third module therefore takes an exponential moving average of the gate and uses it to set the style weight. Alpha is 0.01, the maximum weight is 0.5, and the average starts at zero. This creates a self-paced increase in imitation as the policy becomes more stable. The distinction between the two time scales is important. The fast gate reacts to the current step. The slower weight reflects recent stability. Together, they reduce the influence of style when stability is poor and allow stronger imitation when the policy can support it.

## Slide 7: Simulation protocol

Suggested time: 45 seconds.

We evaluate a self-developed biped that is 1.75 meters tall, weighs 60.9 kilograms, and has twelve actuated leg joints. Training uses PPO in Isaac Sim with the PhysX backend and approximately four thousand parallel environments. The policy trains on flat terrain with randomly sampled velocity commands. Evaluation transfers the policy to MuJoCo without additional training and uses five random seeds. We compare Pure RL, standard AMP, and SG-AMP under flat walking, stronger external pushes, and unseen rough terrain. Every method survives nominal flat walking, so that condition does not distinguish robustness. The more informative comparisons are the two stress tests and the contact and motion-quality metrics.

## Slide 8: Robustness improves in both stress tests

Suggested time: 65 seconds.

This slide shows the central robustness result. The bars report the mean survival rates from Table One. Under stronger external pushes, standard AMP achieves 46.2 percent survival, while SG-AMP reaches 73.6 percent. That is an increase of 27.4 percentage points, or approximately 59.3 percent relative to AMP. On unseen rough terrain, survival increases from 21.8 percent to 38.6 percent. That is 16.8 percentage points, or about 77.1 percent relative improvement. Pure RL is also below SG-AMP in both protocols. The full standard deviations are provided in the backup table. These results support improved robustness in the tested simulation conditions. They do not imply that rough-terrain locomotion is solved: survival remains below forty percent in this challenging protocol.

## Slide 9: Lower jitter and more planar contact

Suggested time: 55 seconds.

The improvements also appear in motion and contact quality. On the left, spectral energy decreases from 2.48 to 1.65 times ten to the minus three compared with standard AMP, a reduction of approximately 33.5 percent. Compared with Pure RL, the reduction is 63.5 percent. Keeping these comparison baselines explicit is important. On the right, the paper’s contact analysis shows the contact model, standard AMP, vanilla PPO, and SG-AMP. The reported planar-contact ratio increases from 46.8 percent for AMP to 85.4 percent for SG-AMP. This metric describes the portion of stance without ankle inversion or eversion tilting. The qualitative images and quantitative metrics together indicate smoother motion and more consistent flat-foot contact in these tests.

## Slide 10: Each module contributes to performance

Suggested time: 65 seconds.

The ablation study removes one component at a time. Without the CoP gate, push survival drops from 73.6 to 51.2 percent. Planar contact also falls sharply, from 85.4 to 49.6 percent. This links the gate to contact quality as well as robustness. Without phase conditioning, spectral energy rises from 1.65 to 3.08 times ten to the minus three, an increase of 86.7 percent. That is consistent with phase information reducing high-frequency fluctuations. Finally, fixing the style weight reduces push survival to 61.5 percent and rough-terrain survival to 30.5 percent. This supports the benefit of adapting imitation intensity. The full method achieves the best mean values in this table. These ablations support complementary roles, although they do not isolate every possible interaction among the components.

## Slide 11: The gate attenuates style more under stress

Suggested time: 40 seconds.

We can also inspect when the gate reduces the style signal. The paper reports the proportion of stance for which the gate is below 0.8. That fraction is 8.4 percent during flat walking, 35.6 percent under pushes, and 44.2 percent on rough terrain. These are attenuation fractions, not failure rates. The pattern is consistent with the intended mechanism: difficult contacts lead to more frequent reductions in imitation strength. This provides a useful behavioral explanation for the robustness results, beyond the final survival numbers alone.

## Slide 12: Conclusion

Suggested time: 35 seconds.

To conclude, SG-AMP makes motion imitation responsive to stability. The CoP gate improves contact behavior, phase conditioning reduces jitter, and the adaptive style weight helps balance imitation with locomotion. Together, these changes improve robustness in our simulation benchmarks while preserving useful motion style. The current study is limited to simulation on one biped platform. Our next steps are hardware transfer and incorporating perception for more varied terrain. Thank you for your attention. I welcome your questions.

## Before the conference: manuscript data checks

以下不一致来自论文原稿。PPT 统一采用 Table I，未修改论文文件。正式报告前建议与原始实验记录核对。

| Item | Conflicting manuscript text | Value used in this deck |
|---|---|---|
| SG-AMP rough survival | Abstract: 0.486; Table I: 0.386 ± 0.035 | 0.386 ± 0.035 |
| Rough improvement over AMP | Abstract and results: +122.9% | +77.1% relative, or +16.8 percentage points, using 0.386 versus 0.218 |
| G3 ablation rough survival | Results paragraph: 0.352; Table I: 0.305 ± 0.039 | 0.305 ± 0.039 |
| Spectral-energy reduction | Abstract: 63.5%, comparison baseline not stated | 33.5% versus AMP; 63.5% versus Pure RL |
| Controller and step | 50 Hz control and 0.005 s step appear together | Both omitted from the main slides pending clarification of simulation versus control time step |
| CoP implementation | Element-level force equation and per-foot net-force implementation description | Main slides explain the mathematical idea; no claim that simulator reconstruction is fully documented |

All chart bars show means. Backup A1 gives the full mean ± standard deviation for five seeds. No p-values or formal safety guarantees are claimed. All experimental evidence is simulation only.

## Suggested Q&A answers

**Does the gate guarantee that the robot will not fall?** No. It is a reward modulation mechanism using a contact-based stability proxy.

**Is this sim-to-real?** The reported transfer is sim-to-sim, from Isaac Sim / PhysX to MuJoCo. Hardware transfer is future work.

**Why use both a gate and an adaptive weight?** The gate responds to the current step, while the moving average adjusts the overall imitation intensity based on recent stability.

**How is phase defined?** A commanded-frequency clock advances the phase, heel-strike events align it, and both reference and policy data use the same convention.

**Which data are needed?** Contact forces, foot geometry, proprioceptive state transitions, and reference motion with phase labels.

**Which module has the strongest contact effect?** In Table I, removing G1 reduces planar contact from 85.4% to 49.6%. Removing G2 has the largest adverse effect on spectral energy.

## Delivery tips

Use PowerPoint Presenter View to see the script and timing. End the main talk at slide 12. Backup A1 and A2 are hidden so that the slide show remains on the conclusion. They can be opened from the slide navigator during questions.

Pronunciation: CoP, “C-O-P”; PPO, “P-P-O”; MuJoCo, “moo-JOH-koh”; sigmoid, “SIG-moyd”.
