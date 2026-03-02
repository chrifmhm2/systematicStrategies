class TransactionCostModel:
    """
    Models the cost of executing a trade.

    Total cost = commission + slippage, applied as basis points of trade value.
    A minimum commission floor is applied per trade.

    Parameters
    ----------
    commission_bps : float
        Broker commission in basis points (1 bps = 0.01%). Default 10 bps.
    slippage_bps : float
        Market impact / bid-ask spread in basis points. Default 5 bps.
    min_commission : float
        Minimum cost per trade in currency units. Default $1.00.
    """

    def __init__(
        self,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        min_commission: float = 1.0,
    ) -> None:
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.min_commission = min_commission

    def compute_cost(self, trade_value: float) -> float:
        """
        Compute the total cost for a trade.

        Parameters
        ----------
        trade_value : float
            Signed or unsigned dollar value of the trade. Cost is based on |trade_value|.

        Returns
        -------
        float
            Total transaction cost in currency units (≥ min_commission).
        """
        cost = abs(trade_value) * (self.commission_bps + self.slippage_bps) / 10_000
        return max(cost, self.min_commission)
