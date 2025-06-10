from zigzag.classes.stages.Stage import Stage
from typing import Callable, List

## Class that calculates the carbon footprint of the accelerator
class CarbonStage(Stage):

    ## The class constructor
    # Initialize the CarbonStage
    # @param list_of_callables (List[Callable]): List of substages to call.
    def __init__(
        self,
        list_of_callables: List[Callable],
        *,
        ci_operational: float,  # in g, CO2/pJ
        ci_embodied: float,  # in g, CO2/mm2
        chip_yield: float,
        chip_area: float = 0.0,  # in mm^2
        tclk: float = 1.0,  # in ns
        chip_lifetime: float = 3,  # in years
        **kwargs
    ):
        super().__init__(list_of_callables, **kwargs)
        self.ci_operational = ci_operational
        self.ci_embodied = ci_embodied
        self.chip_yield = chip_yield
        self.chip_area = chip_area
        self.tclk = tclk
        self.chip_lifetime = chip_lifetime

    def run(self):
        kwargs = self.kwargs.copy()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for cme, extra_info in sub_stage.run():
            energy_operational = cme.energy_total
            latency_in_cycle = cme.latency_total2
            try:
                chip_area = cme.area_total
            except AttributeError:
                chip_area = self.chip_area
            try:
                tclk = cme.tclk
            except AttributeError:
                tclk = self.tclk
            cme.carbon_footprint = self.ci_embodied * chip_area * self.chip_yield * (tclk * chip_area)/self.chip_lifetime + self.ci_operational * energy_operational
            yield (cme, extra_info)
