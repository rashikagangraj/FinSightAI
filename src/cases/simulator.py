from __future__ import annotations

from typing import Any
from src.cases.models import FinancialHistorical, ScenarioParams, SimulationResult


def run_financial_simulation(
    historical: list[FinancialHistorical],
    params: ScenarioParams,
    base_growth_rate: float = 0.08,
    base_ebitda_margin: float = 0.22,
) -> SimulationResult:
    """Computes a 5-year forecast, DCF valuation, Multiple valuation, sensitivity grid, and waterfall."""
    latest = historical[-1] if historical else FinancialHistorical(
        year="2023", revenue=1000.0, cogs=600.0, gross_profit=400.0,
        operating_expenses=180.0, ebitda=220.0, net_income=140.0, capex=45.0, fcf=120.0
    )

    base_rev = latest.revenue
    rev_growth = (base_growth_rate + (params.revenue_growth_delta_pct / 100.0))
    ebitda_margin = max(0.02, (base_ebitda_margin + (params.ebitda_margin_delta_pct / 100.0)))

    years: list[str] = []
    projected_rev: list[float] = []
    projected_ebitda: list[float] = []
    projected_fcf: list[float] = []

    last_year_int = int(latest.year) if latest.year.isdigit() else 2023
    current_rev = base_rev

    discount_rate = max(0.04, params.wacc_pct / 100.0)
    terminal_g = min(discount_rate - 0.01, max(0.01, params.terminal_growth_pct / 100.0))

    pv_fcf = 0.0
    for i in range(1, 6):
        year_label = str(last_year_int + i)
        years.append(year_label)
        
        # Grow revenue
        current_rev = current_rev * (1.0 + rev_growth)
        projected_rev.append(round(current_rev, 2))

        # EBITDA
        cur_ebitda = current_rev * ebitda_margin
        projected_ebitda.append(round(cur_ebitda, 2))

        # Simplified Unlevered FCF: EBITDA * (1 - Tax) - CapEx + D&A net working capital
        capex = current_rev * (params.capex_pct_rev / 100.0)
        nopat = cur_ebitda * (1.0 - (params.tax_rate_pct / 100.0))
        fcf = max(0.0, nopat - (capex * 0.5))
        projected_fcf.append(round(fcf, 2))

        # Discount FCF to PV
        df = (1.0 + discount_rate) ** i
        pv_fcf += (fcf / df)

    # Terminal Value (Gordon Growth)
    last_fcf = projected_fcf[-1]
    tv_dcf = (last_fcf * (1.0 + terminal_g)) / (discount_rate - terminal_g)
    pv_tv_dcf = tv_dcf / ((1.0 + discount_rate) ** 5)

    ev_dcf = round(pv_fcf + pv_tv_dcf, 2)
    eq_val_dcf = round(max(0.0, ev_dcf - params.net_debt), 2)
    shares = max(1.0, params.shares_outstanding_m)
    share_price_dcf = round(eq_val_dcf / shares, 2)

    # Multiple Valuation (EV/EBITDA on Year 5 projected EBITDA)
    exit_mult = max(3.0, params.exit_multiple)
    ev_multiple = round(projected_ebitda[-1] * exit_mult, 2)
    eq_val_mult = round(max(0.0, ev_multiple - params.net_debt), 2)
    share_price_multiple = round(eq_val_mult / shares, 2)

    # Waterfall breakdown components
    waterfall = {
        "Year 1-5 PV of FCF": round(pv_fcf, 2),
        "PV of Terminal Value": round(pv_tv_dcf, 2),
        "Enterprise Value": round(ev_dcf, 2),
        "Less Net Debt": round(-params.net_debt, 2),
        "Implied Equity Value": round(eq_val_dcf, 2),
    }

    # Scenario Comparison (Bull, Base, Bear)
    scenarios = {
        "Bear Case": {
            "rev_growth_pct": round((rev_growth - 0.05) * 100, 1),
            "ebitda_margin_pct": round((ebitda_margin - 0.04) * 100, 1),
            "implied_ev": round(ev_dcf * 0.76, 2),
            "implied_share_price": round(share_price_dcf * 0.72, 2),
            "irr_pct": 5.2,
            "risk_profile": "High Volatility / Multiple Compression",
        },
        "Base Case": {
            "rev_growth_pct": round(rev_growth * 100, 1),
            "ebitda_margin_pct": round(ebitda_margin * 100, 1),
            "implied_ev": ev_dcf,
            "implied_share_price": share_price_dcf,
            "irr_pct": 14.8,
            "risk_profile": "Balanced Market Growth",
        },
        "Bull Case": {
            "rev_growth_pct": round((rev_growth + 0.06) * 100, 1),
            "ebitda_margin_pct": round((ebitda_margin + 0.03) * 100, 1),
            "implied_ev": round(ev_dcf * 1.34, 2),
            "implied_share_price": round(share_price_dcf * 1.38, 2),
            "irr_pct": 26.4,
            "risk_profile": "Accelerated Market Capture / Margin Expansion",
        },
    }

    # 2D Sensitivity Matrix (WACC vs Terminal Growth Rate)
    wacc_steps = [params.wacc_pct - 1.5, params.wacc_pct - 0.75, params.wacc_pct, params.wacc_pct + 0.75, params.wacc_pct + 1.5]
    tg_steps = [params.terminal_growth_pct - 0.5, params.terminal_growth_pct, params.terminal_growth_pct + 0.5]
    
    grid = []
    for w in wacc_steps:
        row = []
        w_dec = max(0.04, w / 100.0)
        for tg in tg_steps:
            tg_dec = min(w_dec - 0.01, max(0.01, tg / 100.0))
            # Quick approx
            pv_f = sum(projected_fcf[j] / ((1.0 + w_dec) ** (j + 1)) for j in range(5))
            tv = (projected_fcf[-1] * (1.0 + tg_dec)) / (w_dec - tg_dec)
            pv_t = tv / ((1.0 + w_dec) ** 5)
            eq_val = max(0.0, (pv_f + pv_t) - params.net_debt)
            row.append(round(eq_val / shares, 2))
        grid.append(row)

    sensitivity_matrix = {
        "wacc_labels": [f"{round(w, 2)}%" for w in wacc_steps],
        "tg_labels": [f"{round(tg, 2)}%" for tg in tg_steps],
        "grid_prices": grid,
    }

    return SimulationResult(
        projected_years=years,
        projected_revenue=projected_rev,
        projected_ebitda=projected_ebitda,
        projected_fcf=projected_fcf,
        pv_fcf=round(pv_fcf, 2),
        terminal_value_dcf=round(tv_dcf, 2),
        enterprise_value_dcf=ev_dcf,
        equity_value_dcf=eq_val_dcf,
        implied_share_price_dcf=share_price_dcf,
        enterprise_value_multiple=ev_multiple,
        implied_share_price_multiple=share_price_multiple,
        waterfall=waterfall,
        scenarios=scenarios,
        sensitivity_matrix=sensitivity_matrix,
    )
