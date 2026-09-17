# Product MtM — Dev Log

Design reasoning, open limitations, and unresolved thinking behind specific modeling choices in
`Product MtM/` — kept separate from `README.md`'s reference material (what things are and how to
run them) so that material stays terse. Written in first person, as the thinking actually
happened; not all of it is resolved yet.

## Why no coupons?

Frankly, I do not yet have the understanding or capability to implement this within the historical outputs. Generating coupons and providing a theoretical value would necessitate actual historical options parsing - I do not know how to necessarily do that in a way I could defend. Furthermore, all coupons are subject to intrinsic fees from both buy-side and sell-side, I do not know what sell-side conventions are and I do not necessarily want to guess and provide a misleading figure. Therefore, I took the decision to not implement them within this project. A keen reader can cross-check the coupon % p.a. from actual issuers. 

## The Zero Coupon Bond Assumption

Another theoretical assumption that was taken was with ZCB's. I do not have the capability to access, find and use historical ZCB's (nor know whether issuer chose to use a full-term ZCB or sequential). Therefore, a rather crude approximation was undertaken to construct a synthetic ZCB. 

In short, the discrete compounding formula (first used) was a: \(\text{Principal} / (1 + \text{SOFR} + \text{spread})^T\). This has since been replaced with continuous compounding after realising the inconsistency, \(\text{Principal} \times e^{-(\text{SOFR} + \text{spread}) \times T}\), to match QuantLib's own `FlatForward` term structures (built with `ql.Continuous` compounding) - the option legs were always discounted continuously inside QuantLib, so the ZCB leg using discrete compounding was an internal inconsistency between the two legs of the same note, not just a separate simplification.

The spread utilised here is Goldman Sachs 1y CDS (Investing.com : 26.75 bps) - issuer credit spread, principal leg only which is obviously a strong assumption. Another strong assumption, perhaps the strongest, undertaken was "SOFR", with the actual SOFR wording being misleading. At first, I assumed a flat risk free rate of conventional 4% for ease and model simplicity. This has now been updated to use the FRED 1-year Treasury Constant Maturity Rate (DGS1) instead of SOFR as SOFR would need to be compounded for the conventional 1 year tenors, as well as being a backwards-looking estimator. I am not sure how accurate/defensible it would be use this sort of backwards-looking rate so therefore I chose a DGS1 proxy.

## Volatility Assumptions

This will be discussed further below but one of the bigger issues is data availability in terms of implied volatility. The underlying volatility used for individual underlyings is historical volatility as I could not locate free, accessible, implied volatility. Therefore, these models may have systematic under/over estimation issues as they are unable to capture the smile/smirk features. 

  Example: assume historical volatility is at x%, however underlying earnings are announced to be in the next week with mixed consensus - therefore, implied volatility would then increase to be above the historical volatility, therefore increasing the premiums and changing the option dynamics based on market conditions. For example, a smile/smirk may not be "picked up" and therefore the actual MtM values may be misleading as the volatility proxy is backwards-looking rather than forwards. 

#### Is there to fix the Volatility Assumption?

Data availability is the main constraint. If I am able to locate a substantiated, reliable source for historical implied volatility this project will be updated with those numbers. 

## Correlation within the Multi-X Structure

This structure proved more difficult. The initial assumption, from my fundamental reading, is that correlation drives up coupon rates due to a higher risk likelihood. If a market/sector is experiencing a sell-off, or vice versa, correlated names are, by definition, going to move in tandem. Therefore, a "look-back" window was implemented on the multiple underlyings - i.e. comparing their daily logarithmic returns over a specified window (2y taken as an arbitrary period) and then computing an actual sample correlation matrix through Pearson sample correlation (standard, simple estimator). 

However, a limitation of this approach is that the correlation matrix is taken as a static, rather than dynamic measure. Therefore, this exposes the product to a rather famous market stress response of "correlation goes to 1 in a crash". The static nature of the correlation matrix does not capture regime-dependence. 

#### Pearson Simplicity and Static Assumptions

As of the moment, I have chosen to use a simplistic correlation estimator. I am aware that it has the tendency to favour linear, Gaussian relationships and can underestimate/be distorted by leptokurtic, non-normal, abnormal data. As this project involves I will implement a number of standard econometrics tools to evaluate the produced correlation matrix (Jarque-Bera, Spearman, etc.), however I have chosen to leave it as of the current moment. 

As this project evolves, I will include a more dynamic measure of correlation - i.e. extending the look-back window to include dates that are within the products life-time and finding a way to create regime-dependency. However, this has not been implemented as of the current moment and I will need to find a way to make a defensible correlator. 
  Issue:
    Assume lookback period 2y, 2 underlyings with correlation 0.6 (arbitrary), model is extended to capture note lifetime and upon first weeks of existence significant market downturn cause symmetric movements within underlying daily logarithmic prices. Despite the significant similarity the correlation matrix is unlikely to experience a massive "inflation" of correlation due to the large sample dynamics it would undergo and therefore the true correlation figure is unlikely to be the historically produced figure. 
  My thinking:
    - Implement a sort of Moving Average (MA) or Exponentially Moving Average (EMA) correlation matrix? 
    - DCC-ARMA-GARCH would probably be the most defensible way to create this correlation figure. The Dynamic Conditional Correlation would be able to capture dynamic, varying correlations within underlyings that may not be serially uncorrelated along the diagonal matrix. 
      Implementation of this will occur at a later point after I understand the DCC-GARCH nature better. My main experience is with univariate GARCH-family models and I have not worked with multivariate models. I want to have a solid baseline of fundamental understanding before approaching a multivariate model and perhaps directing/implementing inconsistencies within the existing code. 

---

See [`README.md`](README.md) for the reference guide (layout, terminology, scope and limitations)
and [`QUANTLIB.md`](QUANTLIB.md) for the QuantLib pricing mechanics.
