| Model                          |        P2V S/L/All |        V2P S/L/All |        IVP S/L/All | Overall |
|---------------------------------------------------------------------------------------------------------|
| Qwen2.5-VL-7B (base)           |     28.9/43.0/40.9 |     34.2/26.2/27.4 |     10.5/ 2.8/ 4.0 |    24.1 |
| Qwen2.5-VL-7B (trained, ours)  |     28.9/24.3/25.0 |     63.2/37.4/41.3 |     73.7/58.9/61.1 |    42.5 |
| GPT-5.4 (zero-shot)            |     65.8/75.2/73.8 |     94.7/65.0/69.4 |     47.4/34.1/36.1 |    59.8 |
| Gemini-3.1-Pro (zero-shot)     |     76.3/77.1/77.0 |     89.5/65.4/69.0 |     21.1/15.9/16.7 |    54.2 |
| Grok-4.20 (zero-shot)          |     73.7/72.0/72.2 |     63.2/73.4/71.8 |     15.8/ 8.9/ 9.9 |    51.3 |
| Claude-Opus-4.6 (zero-shot)    |     39.5/57.9/55.2 |     65.8/55.1/56.7 |     28.9/15.0/17.1 |    43.0 |

Episodes evaluated (All n):
  Qwen2.5-VL-7B (base)           P2V=252  V2P=252  IVP=252
  Qwen2.5-VL-7B (trained, ours)  P2V=252  V2P=252  IVP=252
  GPT-5.4 (zero-shot)            P2V=252  V2P=252  IVP=252
  Gemini-3.1-Pro (zero-shot)     P2V=252  V2P=252  IVP=252
  Grok-4.20 (zero-shot)          P2V=252  V2P=252  IVP=252
  Claude-Opus-4.6 (zero-shot)    P2V=252  V2P=252  IVP=252

Test-set turn split: short(<=2)=38  long(>2)=214  total=252
