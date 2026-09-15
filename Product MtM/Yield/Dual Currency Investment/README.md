# Dual Currency Investment (DCI) Historical Approximation

Traces the historical mark-to-market of a **Dual Currency Investment**: a short-dated FX
structure where a principal amount sits in a **deposit currency** money-market deposit while the
investor simultaneously **sells a put option on an alternate currency**, struck at a rate below
today's spot. Defaults to `DEPOSIT_CCY = "USD"` / `ALT_CCY = "EUR"`, but any listed FX pair works.

This is a **deterministic historical approximation** — real historical FX prices, no simulation or
Monte Carlo (same style as the other products in `Product MtM/`).

Mechanically this is the **same structure as the Reverse Convertible** (sibling folder under
`Yield/`), with the alternate currency playing the role of "the stock" and Garman-Kohlhagen (the
FX version of Black-Scholes-Merton) replacing the equity option model. If you understand the RC,
you understand this.

## Mechanics

```
Scenario A: ALT strengthens or stays flat vs. the strike -> put expires worthless
            -> principal stays in DEPOSIT currency
Scenario B: ALT weakens past the strike                  -> bank exercises the put
            -> principal is converted into ALT at the strike rate
```

The put premium the investor collects for taking on scenario B is what enhances the deposit's
yield above a plain money-market rate — that's the whole economic story, identical in spirit to
why a Reverse Convertible's coupon beats a plain bond's.

## The quoting convention (read this first)

FX rates can be quoted in two directions, and it matters which one a strike is measured against.
This file uses:

```
S = how many units of DEPOSIT currency 1 unit of ALT currency costs
```

i.e. for `DEPOSIT=USD, ALT=EUR`, `S` is the standard "EURUSD" market quote (~1.05-1.15) — literally
`price of 1 EUR, in USD`. This is deliberate, not arbitrary: it's the convention **Garman-Kohlhagen
itself requires** (domestic currency = DEPOSIT, foreign currency = ALT, and QuantLib's
`BlackScholesMertonProcess` — same engine the Reverse Convertible uses — already computes exactly
this model once `q` is set to the foreign/ALT short rate), and it makes `S` and `STRIKE` play
**exactly** the role of "stock price" and "strike" in the Reverse Convertible, so the same
replication and code apply unchanged.

Under this convention: ALT strengthening means `S` (USD per EUR) **rises**, so "ALT strengthens or
stays flat vs. strike" = `S_T >= STRIKE`, and "ALT weakens past strike" = `S_T < STRIKE` — the put
is out-of-the-money above the strike and in-the-money below it, same direction as the Reverse
Convertible's put on the stock.

**A note on the opposite convention**: some pairs (USDJPY, USDCHF, USDCAD) are, by market
convention, quoted the other way round — as "how many units of the *other* currency 1 USD buys."
If you're used to thinking in that direction (USD as the numerator), "ALT strengthens" corresponds
to the quoted rate **falling**, i.e. "you want to stay *below* the strike" when looking at it that
way — same real-world economics, just the mirror image of the same number. This file always uses
the DEPOSIT-per-1-ALT direction (`S = USD per EUR`, not `EUR per USD`) so that `STRIKE` and the
Reverse Convertible's `STRIKE` mean the same thing; a worked example:

> `DEPOSIT=USD, ALT=EUR, S0 = 1.0352, STRIKE = 0.95` → strike level = `0.9834` USD per EUR.
> If EUR/USD drifts up to `1.08` by maturity (EUR **strengthened**), `1.08 >= 0.9834`, so the put
> expires worthless — principal stays in USD. If EUR/USD instead fell to `0.95` (EUR **weakened**),
> `0.95 < 0.9834`, so the bank exercises: `Principal_USD` is converted into EUR at the strike rate
> `0.9834`, i.e. the investor receives `Principal_USD / 0.9834` euros — worth
> `Principal_USD * (0.95 / 0.9834) ≈ 96.6%` of the original USD principal at the real closing
> rate. Exactly the Reverse Convertible's `Principal * S_T/Strike` formula, with EUR standing in
> for the stock.

## Replication

```
DCI = Long DEPOSIT-currency balance (pays Principal at maturity)
    - Short Put on ALT (struck at STRIKE, e.g. 95% of S0)
```

**Terminal payoff**: full principal back (in DEPOSIT) if `S_T >= STRIKE * S0`; below it, the investor
receives `Principal / STRIKE` units of ALT instead of DEPOSIT cash — same conversion-ratio logic as
the Reverse Convertible's physical share delivery (see that folder's README for the full
derivation of why the quantity is `1/STRIKE`, not `1x`). In value terms:
`Principal * min(S_T/STRIKE, 1)`.

## Discounting and option pricing: two different rates for two different risks

- **Deposit leg** — discounted at `Principal / (1 + DEPOSIT_RATE + ISSUER_CDS_SPREAD)^T`. The
  investor is exposed to the deposit-taking bank's own default risk, same reasoning as the RC's
  ZCB leg — reuses the same real, sourced Goldman Sachs 5y CDS spread (53.08 bps) as an
  illustrative counterparty; replace it if modeling a different bank.
- **Put leg** — priced via **QuantLib** (`AnalyticEuropeanEngine`, closed-form
  Black-Scholes-Merton with `q = ALT_RATE`), at `DEPOSIT_RATE` alone, no credit spread — same
  reasoning as the RC: a bank prices/hedges the option on standard derivative terms, not its own
  funding curve. With `q` set to the ALT currency's own short rate, this process **is**
  Garman-Kohlhagen — holding a foreign currency continuously earns its own risk-free rate, the
  exact same mathematical role a continuous dividend yield plays for a stock.

## Rates: `DEPOSIT_RATE` and `ALT_RATE` are modeling assumptions, not fetched data

Just like `RISK_FREE_RATE` in the Reverse Convertible/FCN/Discount Certificate scripts elsewhere
in this repo, `DEPOSIT_RATE` (4%, "USD rate") and `ALT_RATE` (2%, "EUR rate") are **flat, clearly
labeled assumptions you set**, not numbers fetched from a live source. There is no single generic
per-currency short-rate feed wired up here (unlike the dividend-yield fetch in the RC, which does
pull real data) — set these to whatever real short rates you want to model for your two currencies
(e.g. SOFR/EFFR for USD, the ECB deposit facility rate for EUR, SONIA for GBP, TONA for JPY).

## Volatility

No generic FX-implied-volatility source exists for an arbitrary currency pair the way VIX serves
SPX for the equity products in this repo, so this file uses the pair's own **realized historical
volatility** as the pricing input — a real, computed number, not invented, but a genuine
simplification versus a true FX implied-vol surface (which would require a paid data feed). This
is the same category of simplification as everywhere else a flat assumption stands in for a richer
real input in this repo (e.g. the flat `RISK_FREE_RATE`).

## Product terms

| Term | Default | Meaning |
|---|---|---|
| `STRIKE` | 95% of entry level | Short put strike / conversion level |
| `DEPOSIT_CCY` / `ALT_CCY` | USD / EUR | Deposit currency / currency the put is sold on |
| `DEPOSIT_RATE` | 4% (flat) | DEPOSIT currency short-term rate assumption |
| `ALT_RATE` | 2% (flat) | ALT currency short-term rate assumption |
| `ISSUER_CDS_SPREAD` | 53.08 bps | Goldman Sachs 5y CDS - issuer credit spread, deposit leg only |
| `ENTRY_DATE` / `TENOR` | 2025-01-02 / 3 months | The historical window - DCIs are typically short-dated (days to a few months), unlike the 1-year convention used elsewhere in this repo |

Edit the constants at the top of `DCI.py` to reprice the note, change the currency pair, or change
the window.

## Greeks

Same structure as the Reverse Convertible:

- **Delta** comes entirely from the short put (`-(1/STRIKE) * put_delta`) - the deposit leg has no
  FX sensitivity at all.
- **Vega** is `-(1/STRIKE) * put_vega` - short volatility.
- **Rho** combines the deposit's duration sensitivity to `DEPOSIT_RATE` with the put's own rho.
- **Theta** is positive by construction: the deposit "pulls to par" as time passes, and the short
  put decays in the position's favor.

## Output

Running `DCI.py` prints the entry/maturity spot levels, realized FX return, the note's return, the
note's fair value at inception (as % of par), and its Greeks - then saves a chart with the FX
spot path and note payoff on the left axis and the note's fair value on its own right-hand axis,
with a dotted line marking the strike.

## Usage

```
pip install -r ../../../requirements.txt
python3 "DCI.py"
```

For the Greeks ladder (spot varies, tenor/strike/rates/vol held fixed):

```
python3 "Greek Sensitivity.py"
```

Data comes from `yfinance` (Yahoo Finance) - FX tickers are constructed as `f"{ALT_CCY}{DEPOSIT_CCY}=X"`
(verified empirically: Yahoo's `"{CCY1}{CCY2}=X"` convention returns units of `CCY2` per 1
`CCY1`, so this always returns `S` in the DEPOSIT-per-1-ALT convention this file uses throughout).
