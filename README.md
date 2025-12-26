## Motivation
Time series models struggle with regime changes and global shape patterns.

## Idea
Convert recent history into an image (GAF) and use a vision encoder
to extract semantic features (trend, volatility, regime).

## Method
- Numeric Transformer baseline
- CNN over GAF image
- Late fusion

## Results
(table)

## Ablations
Shuffled images remove gains → modality matters.

## When it helps / when it doesn’t
(honest discussion)

## Next steps
- Image-to-text captions
- Cross-attention fusion
- Probabilistic forecasting
