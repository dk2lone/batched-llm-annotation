# X1R4 A8 full-grid lock

The CPU-only lock fixes all sixteen combinations of four models and four datasets at 8,192 items per cell. It used no extension joint output and launched no GPU work.

The grid contains 24,576 joint calls, 393,216 exact-context calls, and 131,072 solo calls, for 548,864 calls total. The same ordered 8,192 dataset items are used across all four models and all conditions.

The primary analysis is joint-minus-exact error covariance over the complete grid, with 4,999 replay draws, sixteen-cell Holm correction, equal model/task weight, model and task margins, interaction terms, leave-one-out checks, and four frozen malformed-output treatments. Accuracy and parse rate cannot remove a cell.

This lock does not authorize a GPU start. A separate technical-only forecast must clear prior + 1.6 × complete-block base forecast + 10 <= 80, followed by a measured reforecast after each model's first 1,000 A8 attempts. Scientific outcomes cannot alter the fixed order or 8,192-item count.
