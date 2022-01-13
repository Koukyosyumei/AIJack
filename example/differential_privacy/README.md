## Usage

## Comparison with Opacus

|                               | `get_epsilon`     | `get_noise_multiplier` |
| ----------------------------- | ----------------- | ---------------------- |
| Opacus                        | 19.3 ms ± 3.47 ms | 208 ms ± 14.6 ms       |
| AIJack with SGM               | 816 µs ± 180 µs   | 8.25 ms ± 1.18 ms      |
| AIJack with tight upper bound | 1.33 ms ± 310 µs  | 15.7 ms ± 7.24 ms      |
