import logging
from typing import Dict, List, Tuple
from math import ceil, log2
import numpy as np

from zigzag.classes.mapping.combined_mapping import Mapping
from zigzag.classes.mapping.combined_mapping import FourWayDataMoving
from zigzag.utils import pickle_deepcopy

logger = logging.getLogger(__name__)


##  Class that collects all the data transfer rate (periodic) information for each DTL (data transfer link).
class PortActivity:

    ## The class constructor
    # @param real_cycle     (int)     Within each period, the actual number of cycles used for transferring the amount of data, depended on the memory bw and the data amount to be transferred at that memory level.
    # @param allowed_cycle  (int)
    # @param period         (int)     The turnaround cycle at that memory level, which equals to the product of all the temporal loops of current and below memory level.
    # @param period_count   (int)     The total number of period across the whole NN layer computation.
    # @param layer_op       (str)
    # @param mem_lv         (int)
    # @param mov_dir        (str)
    def __init__(
            self,
            real_cycle: int,
            allowed_cycle: int,
            period: int,
            period_count: int,
            layer_op: str,
            mem_lv: int,
            mov_dir: str,
    ):
        ## Within each period, the actual number of cycles used for transferring the amount of data, depended on the memory bw and the data amount to be transferred at that memory level.
        self.real_cycle = real_cycle
        self.allowed_cycle = allowed_cycle
        ## The turnaround cycle at that memory level, which equals to the product of all the temporal loops of current and below memory level.
        self.period = period
        ## The total number of period across the whole NN layer computation.
        self.period_count = period_count
        self.served_op_lv_dir = (layer_op, mem_lv, mov_dir)
        """ stalling (+) or slacking (-) cycle in one period """
        self.SS_per_period = real_cycle - allowed_cycle
        """ stalling (+) or slacking (-) cycle in total computation """
        self.SS = (real_cycle - allowed_cycle) * (period_count - 1)
        """ total memory updating window allowed """
        self.MUW = allowed_cycle * (period_count - 1)

    def __str__(self):
        return str(self.served_op_lv_dir)

    def __repr__(self):
        return str(self.served_op_lv_dir)

    def __eq__(self, other) -> bool:
        return str(self.served_op_lv_dir) == other

    def __hash__(self):
        return str(self.served_op_lv_dir)


## Class that collects all the data transfer rate information for each DTL (data transfer link).
class PortBeginOrEndActivity:

    ## The class constructor
    # @param real_cycle     (int)   the actual number of cycles used for transferring the amount of data,
    #                               depended on the memory bw and the data amount to be transferred at that memory level
    # @param data_in_charge (int)   one-period data transfer amount (bit)
    # @param mem_bw         (int)   bit/cycle
    # @param layer_op       (str)
    # @param mem_lv         (int)
    # @param mov_dir        (str)   data moving direction
    def __init__(
            self,
            real_cycle: int,
            data_in_charge: int,
            mem_bw: int,
            layer_op: str,
            mem_lv: int,
            mov_dir: str,
    ):
        ## the actual number of cycles used for transferring the amount of data,
        # depended on the memory bw and the data amount to be transferred at that memory level
        self.real_cycle = real_cycle
        ## one-period data transfer amount (bit)
        self.data_in_charge = data_in_charge
        ## bit/cycle
        self.mem_bw = mem_bw
        self.served_op_lv_dir = (layer_op, mem_lv, mov_dir)

    def __str__(self):
        return str(self.served_op_lv_dir)

    def __repr__(self):
        return str(self.served_op_lv_dir)


## Given a certain operand's storage level (for example (A,1): operand A's 1st memory level),
# return a list of the rest operand's storage levels that share physical memory with the former one (A1)
# @param mem_op
# @param mem_lv
# @param memory_sharing_list
# @return mem_share_grp
def get_shared_mem_list(mem_op, mem_lv, memory_sharing_list) -> List[Tuple]:
    for mem_share_group in memory_sharing_list:
        mem_share_grp = list(mem_share_group.items())
        mem_target = (mem_op, mem_lv)
        if mem_target in mem_share_grp:
            return mem_share_grp


## Generate the integer spatial mapping from fractional spatial mapping (due to greedy mapping support).
# Later the fractional one is used for calculating energy, and the integer one is used for calculating latency
# @param spatial_mapping
# @return spatial_mapping_int
def spatial_mapping_fractional_to_int(spatial_mapping: Dict):
    spatial_mapping_int = pickle_deepcopy(spatial_mapping)
    for op, su_all_lv in spatial_mapping.items():
        if not su_all_lv:
            continue
        for lv, su_one_level in enumerate(su_all_lv):
            for idx, su in enumerate(su_one_level):
                if type(su[1]) != int:
                    spatial_mapping_int[op][lv][idx] = (su[0], ceil(su[1]))

    return spatial_mapping_int


##  This function calculates the union length of all the share-port MUW (memory updating window).
#   The following encoding has to be used:
#   - 'P' for single period length
#   - 'A' for allowed MUW per period
#   - 'PC' for period count within the whole layer computation
#
#   Pre-process the port_duty_list to generate input_dict, which looks like:
#   - input_dict = {'O1': {'P': 3, 'A': 1, 'PC': 8}, 'O2': {'P': 6, 'A': 2, 'PC': 4}, 'O3': {'P': 12, 'A': 4, 'PC': 2}}
#
#   @param port_duty_list List of port activity objects
#   @reutrn
def calc_MUW_union(port_duty_list):
    input_dict = {}
    for port_duty in port_duty_list:
        """as long as one of the port duty can make use of the whole computation time, the MUW union is set to
        the whole computation time"""
        if port_duty.period == port_duty.allowed_cycle:
            return port_duty.period * port_duty.period_count
        key = str(port_duty.served_op_lv_dir)
        input_dict[key] = {
            "P": port_duty.period,
            "A": port_duty.allowed_cycle,
            "PC": port_duty.period_count,
        }

    max_period = 0
    max_period_operand = None
    for op in input_dict:
        if input_dict[op]["P"] > max_period:
            max_period = input_dict[op]["P"]
            max_period_operand = op

    indicators = np.zeros((len(input_dict), max_period), dtype=np.int8)
    for i, op in enumerate(input_dict):
        """reshape to period of this operand"""
        indicators_reshape = indicators.reshape(
            (len(input_dict), -1, input_dict[op]["P"])
        )
        """ fill in first few time units as used """
        indicators_reshape[i, :, : input_dict[op]["A"]] = 1

    union = max_period - (~indicators.any(0)).sum(dtype=np.uint64)

    # take sum across operands => how many operand need memory for every time unit
    # Subtract 1 => number of stalls
    # Clip by 0 (-1 is not -1 stall)
    # Sum across time units (only remaining axis)
    # stall = (indicators.sum(0, dtype=np.int8) - 1).clip(min=0).sum()

    """ Multiply with number of periods of largest period (as it was normalized to largest period) """
    return union * input_dict[max_period_operand]["PC"]


## Class that stores inputs and runs them through the zigzag cost model.
#
# Initialize the cost model evaluation with the following inputs:
# - accelerator: the accelerator that includes the core on which to run the layer
# - layer: the layer to run
# - spatial_mapping: the spatial mapping
# - temporal_mapping: the temporal mapping
#
# From these parameters, the following attributes are computed:
# * core: The core on which the layer is ran. This should be specified in the LayerNode attributes.
# * mapping: The combined spatial and temporal mapping object where access patterns are computed.
#
# The following cost model attributes are also initialized:
# - energy_breakdown: The energy breakdown for all operands
# - energy: The total energy
#
# After initialization, the cost model evaluation is run.
class CostModelEvaluation:

    ## The class constructor
    # After initialization, the cost model evaluation is run
    # @param accelerator the accelerator that includes the core on which to run the
    # @param layer the layer to run
    # @param spatial_mapping the spatial mapping
    # @param temporal_mapping the temporal mapping
    # @param access_same_data_considered_as_no_access (optional)
    def __init__(
            self,
            *,
            accelerator,
            layer,
            spatial_mapping,
            spatial_mapping_int,
            temporal_mapping,
            access_same_data_considered_as_no_access=True,
    ):
        self.accelerator = accelerator
        self.layer = layer
        self.spatial_mapping = spatial_mapping
        self.spatial_mapping_int = spatial_mapping_int  # the original spatial mapping without decimal
        self.temporal_mapping = temporal_mapping
        self.access_same_data_considered_as_no_access = (
            access_same_data_considered_as_no_access
        )

        self.core_id = layer.core_allocation
        self.mem_level_list = (
            accelerator.get_core(self.core_id).get_memory_hierarchy().mem_level_list
        )
        self.mem_hierarchy_dict = accelerator.get_core(
            self.core_id
        ).get_memory_hierarchy_dict()
        self.mem_size_dict = accelerator.get_core(self.core_id).get_memory_size_dict()
        self.mem_r_bw_dict, self.mem_w_bw_dict = accelerator.get_core(
            self.core_id
        ).get_memory_bw_dict()
        self.mem_r_bw_min_dict, self.mem_w_bw_min_dict = accelerator.get_core(
            self.core_id
        ).get_memory_bw_min_dict()
        self.mem_sharing_list = accelerator.get_core(
            self.core_id
        ).get_memory_sharing_list()
        self.layer_op_to_mem_op = layer.memory_operand_links
        self.mem_op_to_layer_op = dict(
            [(value, key) for key, value in self.layer_op_to_mem_op.items()]
        )

        """ generate the integer spatial mapping from fractional spatial mapping (due to greedy mapping support).
        Later the fractional one is used for calculating energy, and the integer one is used for calculating latency"""
        # self.spatial_mapping_dict_int = spatial_mapping_fractional_to_int(
        #     self.spatial_mapping.mapping_dict_origin
        # )
        self.spatial_mapping_dict_int = self.spatial_mapping_int.mapping_dict_origin

        # For constructing Mapping object,  the last parameter "self.access_same_data_considered_as_no_access" is optional
        self.mapping = Mapping(
            self.accelerator,
            self.spatial_mapping,
            self.temporal_mapping,
            self.layer,
            self.access_same_data_considered_as_no_access,
        )
        self.mapping_int = Mapping(
            self.accelerator,
            self.spatial_mapping_dict_int,
            self.temporal_mapping,
            self.layer,
            self.access_same_data_considered_as_no_access,
        )

        self.active_mem_level = self.mapping.mem_level

        # Run the cost model evaluation
        self.run()

    def __str__(self):
        return f"CostModelEvaluation(layer={self.layer}, core={self.core_id})"

    def __repr__(self):
        return str(self)

    # JSON representation used for saving this object to a json file.
    def __jsonrepr__(self):
        computation_breakdown = {
            "mac_computation": self.ideal_temporal_cycle,
            "mem_stalling": self.SS_comb,
        }
        return {
            "outputs": {
                "memory": {
                    "utilization": self.mem_utili_shared
                    if hasattr(self, "mem_utili_shared")
                    else None,
                    "word_accesses": self.memory_word_access,
                },
                "energy": {
                    "energy_total": self.energy_total,
                    "operational_energy": self.MAC_energy,
                    "memory_energy": self.mem_energy,
                    "energy_breakdown_per_level": self.energy_breakdown,
                    "energy_breakdown_per_level_per_operand": self.energy_breakdown_further,
                },
                "latency": {
                    "data_onloading": self.latency_total1 - self.latency_total0,
                    "computation": self.latency_total0,
                    "data_offloading": self.latency_total2 - self.latency_total1,
                    "computation_breakdown": computation_breakdown,
                },
                "spatial": {
                    "mac_utilization": {
                        "ideal": self.MAC_spatial_utilization,
                        "stalls": self.MAC_utilization0,
                        "stalls_onloading": self.MAC_utilization1,
                        "stalls_onloading_offloading": self.MAC_utilization2,
                    }
                },
            },
            "inputs": {
                "accelerator": self.accelerator,
                "layer": self.layer,
                "spatial_mapping": self.spatial_mapping
                if hasattr(self, "spatial_mapping")
                else None,
                "temporal_mapping": self.temporal_mapping
                if hasattr(self, "temporal_mapping")
                else None,
            },
        }

    ## Simple JSON representation used for saving this object to a simple json file.
    def __simplejsonrepr__(self):
        return {"energy": self.energy_total, "latency": self.latency_total2}

    ## Run the cost model evaluation.
    def run(self):
        # - TODO: Latency calculation
        self.calc_memory_utilization()
        self.calc_memory_word_access()
        self.calc_energy()
        self.calc_latency()

    ## Calculate occupancy for each physical memory based on the mapping.
    def calc_memory_utilization(self):
        # mem_utili_individual: the memory utilization of each operand individually.
        # mem_utili_shared: the memory utilization taking operand memory sharing into consideration.
        mem_utili_individual = {}
        effective_mem_utili_individual = {}
        for layer_op in self.layer.operand_list:
            mem_utili_individual[layer_op] = []
            effective_mem_utili_individual[layer_op] = []
            for mem_lv in range(self.active_mem_level[layer_op]):
                mem_utilization = (
                        self.mapping.data_bit_per_level_unrolled[layer_op][mem_lv + 1]
                        / self.mem_size_dict[self.layer_op_to_mem_op[layer_op]][mem_lv]
                )
                assert mem_utilization <= 1, (
                    f"Operand {layer_op} memory level {mem_lv}'s individual memory utilization is "
                    f"{mem_utilization}, which is larger than 1 "
                    f"(memory level starts from 0)"
                )
                mem_utili_individual[layer_op].append(mem_utilization)

                # if we do not count copied data in parallel memories as effective, what is the utilization then? =>
                effective_mem_utilization = (
                        self.mapping.effective_data_bit[layer_op][mem_lv + 1]
                        / self.mem_size_dict[self.layer_op_to_mem_op[layer_op]][mem_lv]
                )
                effective_mem_utili_individual[layer_op].append(
                    effective_mem_utilization
                )

        mem_utili_shared = pickle_deepcopy(mem_utili_individual)
        effective_mem_utili_shared = pickle_deepcopy(effective_mem_utili_individual)
        for mem_share_dict in self.mem_sharing_list:
            mem_utilization = 0
            effective_mem_utilization = 0
            for mem_op, mem_lv in mem_share_dict.items():
                try:
                    layer_op = self.mem_op_to_layer_op[mem_op]
                except:  # mem to layer op might not contain this mem op (e.g. pooling layer)
                    continue
                mem_utilization += mem_utili_individual[layer_op][mem_lv]
                effective_mem_utilization += effective_mem_utili_individual[layer_op][
                    mem_lv
                ]
            assert mem_utilization <= 1, (
                f"Memory shared by {mem_share_dict} (memory operand, memory level) has shared utilization of "
                f"{mem_utilization}, which is > 1 "
                f"(memory level starts from 0)."
            )
            for mem_op, mem_lv in mem_share_dict.items():
                try:
                    layer_op = self.mem_op_to_layer_op[mem_op]
                except:  # mem to layer op might not contain this mem op (e.g. pooling layer)
                    continue
                mem_utili_shared[layer_op][mem_lv] = mem_utilization
                effective_mem_utili_shared[layer_op][mem_lv] = effective_mem_utilization

        self.mem_utili_individual = mem_utili_individual
        self.mem_utili_shared = mem_utili_shared
        self.effective_mem_utili_individual = effective_mem_utili_individual
        self.effective_mem_utili_shared = effective_mem_utili_shared

    ## Calculates the memory word access based on unit memory's data element move count and the physical memory bw.
    def calc_memory_word_access(self):
        memory_word_access = {}
        for layer_op in self.layer.operand_list:
            memory_word_access[layer_op] = []
            for mem_lv in range(self.mapping.mem_level[layer_op]):
                """wr_in_by_low"""
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.wr_in_by_low
                data_precision = self.mapping.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.wr_in_by_low
                if data_elem_move_per_period == 0 or data_precision == 0:
                    wr_in_by_low = 0
                else:
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.wr_in_by_low
                    max_bw = self.mem_w_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_w_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    if mem_lv > 0:
                        another_side_bw = self.mem_r_bw_dict[
                                              self.layer_op_to_mem_op[layer_op]
                                          ][mem_lv - 1] * (
                                                  self.spatial_mapping.unit_unique[layer_op][mem_lv]
                                                  / self.spatial_mapping.unit_unique[layer_op][mem_lv + 1]
                                          )
                        data_elem_move_per_cycle_in_a_period = min(
                            (another_side_bw / data_precision),
                            (max_bw / data_precision),
                            data_elem_move_per_period,
                        )
                        cycle_in_a_period = ceil(
                            data_elem_move_per_period
                            / data_elem_move_per_cycle_in_a_period
                        )
                    else:
                        data_elem_move_per_cycle_in_a_period = data_elem_move_per_period
                        cycle_in_a_period = 1
                    # wr_in_by_low = (
                    #    ceil(
                    #        (data_elem_move_per_cycle_in_a_period * data_precision)
                    #        / min_bw
                    #    )
                    #    * (min_bw / max_bw)
                    #    * total_period_count
                    #    * cycle_in_a_period
                    #    * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    # )

                    # 2023/06/30, solve the memory access granuarity issue - Jiacong Sun, Linyan Mei
                    # Originally we used the cycle_in_a_period to compute the memory word access.
                    # This neglected the finer-grained memory access possibility (the min_bw, the minimal memory access granuarity, like half-word access).
                    # Now we changed to calculation based on min_bw.
                    wr_in_by_low = (
                            ceil((data_elem_move_per_period * data_precision) / min_bw)
                            * (min_bw / max_bw)
                            * total_period_count
                            * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ rd_out_to_low """
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.rd_out_to_low
                data_precision = self.mapping.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.rd_out_to_low
                if data_elem_move_per_period == 0 or data_precision == 0:
                    rd_out_to_low = 0
                else:
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.rd_out_to_low
                    max_bw = self.mem_r_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_r_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    if mem_lv > 0:
                        another_side_bw = self.mem_w_bw_dict[
                                              self.layer_op_to_mem_op[layer_op]
                                          ][mem_lv - 1] * (
                                                  self.spatial_mapping.unit_unique[layer_op][mem_lv]
                                                  / self.spatial_mapping.unit_unique[layer_op][mem_lv + 1]
                                          )
                        data_elem_move_per_cycle_in_a_period = min(
                            (another_side_bw / data_precision),
                            (max_bw / data_precision),
                            data_elem_move_per_period,
                        )
                        cycle_in_a_period = ceil(
                            data_elem_move_per_period
                            / data_elem_move_per_cycle_in_a_period
                        )
                        # rd_out_to_low = (
                        #    ceil(
                        #        (data_elem_move_per_cycle_in_a_period * data_precision)
                        #        / min_bw
                        #    )
                        #    * (min_bw / max_bw)
                        #    * total_period_count
                        #    * cycle_in_a_period
                        #    * self.mapping.spatial_mapping.unit_count[layer_op][
                        #        mem_lv + 1
                        #    ]
                        # )
                    # else:

                    # 2023/06/30, solve the memory access granuarity issue - Jiacong Sun, Linyan Mei
                    # Originally we used the cycle_in_a_period to compute the memory word access.
                    # This neglected the finer-grained memory access possibility (the min_bw, the minimal memory access granuarity, like half-word access).
                    # Now we changed to calculation based on min_bw.
                    rd_out_to_low = (
                            ceil((data_elem_move_per_period * data_precision) / min_bw)
                            * (min_bw / max_bw)
                            * total_period_count
                            * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ rd_out_to_high """
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.rd_out_to_high
                if data_elem_move_per_period == 0:
                    rd_out_to_high = 0
                else:
                    data_precision = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_precision.rd_out_to_high
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.rd_out_to_high
                    max_bw = self.mem_r_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_r_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    rd_out_to_high = (
                            ceil((data_elem_move_per_period * data_precision) / min_bw)
                            * (min_bw / max_bw)
                            * total_period_count
                            * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ wr_in_by_high """
                data_elem_move_per_period = self.mapping.unit_mem_data_movement[
                    layer_op
                ][mem_lv].data_trans_amount_per_period.wr_in_by_high
                if data_elem_move_per_period == 0:
                    wr_in_by_high = 0
                else:
                    data_precision = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_precision.wr_in_by_high
                    total_period_count = self.mapping.unit_mem_data_movement[layer_op][
                        mem_lv
                    ].data_trans_period_count.wr_in_by_high
                    max_bw = self.mem_w_bw_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    min_bw = self.mem_w_bw_min_dict[self.layer_op_to_mem_op[layer_op]][
                        mem_lv
                    ]
                    wr_in_by_high = (
                            ceil((data_elem_move_per_period * data_precision) / min_bw)
                            * (min_bw / max_bw)
                            * total_period_count
                            * self.mapping.spatial_mapping.unit_count[layer_op][mem_lv + 1]
                    )

                """ All """
                memory_word_access_single = FourWayDataMoving(
                    rd_out_to_low, wr_in_by_low, rd_out_to_high, wr_in_by_high
                )
                memory_word_access[layer_op].append(memory_word_access_single)

        self.memory_word_access = memory_word_access

    ## Calculates the energy cost of this cost model evaluation by calculating the memory reading/writing energy.
    def calc_energy(self):
        # - TODO: Interconnection energy
        self.calc_MAC_energy_cost()
        self.calc_memory_energy_cost()

    ## Calculate the dynamic MAC energy
    def calc_MAC_energy_cost(self, gating_used=True):
        core = self.accelerator.get_core(self.core_id)
        single_MAC_energy = core.operational_array.unit.cost
        if not gating_used:
            # calc activate multipliers
            total_col_counts = self.accelerator.cores[0].operational_array.dimension_sizes[0]
            total_row_counts = self.accelerator.cores[0].operational_array.dimension_sizes[1]
            try:
                active_rows_mapping = self.layer.user_spatial_mapping["D1"]
            except KeyError:  # D1 has no spatial loop
                active_rows_mapping = None
            try:
                active_cols_mapping = self.layer.user_spatial_mapping["D2"]
            except KeyError:  # D2 has no spatial loop
                active_cols_mapping = None

            if active_rows_mapping is None:
                active_row_counts = 1
            elif self.is_nested_tuple(active_rows_mapping):
                active_row_counts = 1
                for ele in active_rows_mapping:
                    active_row_counts *= ele[1]
            else:
                active_row_counts = active_rows_mapping[1]

            if active_cols_mapping is None:
                active_col_counts = 1
            elif self.is_nested_tuple(active_cols_mapping):
                active_col_counts = 1
                for ele in active_cols_mapping:
                    active_col_counts *= ele[1]
            else:
                active_col_counts = active_cols_mapping[1]
            active_macros = self.mapping.spatial_mapping.unit_count["W"][0] / (active_row_counts * active_col_counts)
            macro_activation_counts = self.layer.total_MAC_count / self.mapping.spatial_mapping.unit_count["W"][0]
            # tell if the dataflow is weight stationary or output stationary
            lowest_mem_level = self.accelerator.cores[0].memory_hierarchy.mem_level_list[0]
            served_mem_op = lowest_mem_level.operands[0]
            served_layer_op = self.layer.get_layer_operand(served_mem_op)
            output_layer_op = self.layer.output_operand
            mult_counts = macro_activation_counts * active_macros * \
                          (active_row_counts * active_col_counts)
            if served_layer_op == output_layer_op:  # output stationary
                half_mult_counts = macro_activation_counts * active_macros * \
                          (active_row_counts * total_col_counts + active_col_counts * total_row_counts - 2 * active_row_counts * active_col_counts)
            else:  # weight stationary
                half_mult_counts = macro_activation_counts * active_macros * \
                          (active_row_counts * (total_col_counts - active_col_counts))
            mult_energy = single_MAC_energy * mult_counts + 0.5 * single_MAC_energy * half_mult_counts
        else:
            mult_energy = single_MAC_energy * self.layer.total_MAC_count
        adders_energy = self.calc_adders_energy()
        self.MAC_energy = adders_energy + mult_energy

    def calc_adders_energy(self):
        if self.layer.type not in ["matmul", "gemm", "conv"]:  # debugging branch
            breakpoint()
        if self.layer.type == "matmul":
            adder_tree_energy = 0
            return adder_tree_energy
        # tell if the dataflow is weight stationary or output stationary
        # add energy for weight stationary dataflow
        lowest_mem_level = self.accelerator.cores[0].memory_hierarchy.mem_level_list[0]
        served_mem_op = lowest_mem_level.operands[0]
        served_layer_op = self.layer.get_layer_operand(served_mem_op)
        output_layer_op = self.layer.output_operand
        output_mem_op = self.layer.memory_operand_links[output_layer_op]
        act_layer_op = [op for op in self.layer.input_operand_source.keys()][0]
        if served_layer_op == output_layer_op:  # output stationary
            adder_tree_energy = 0
        else:  # weight stationary
            # calc number of macro activation times
            macro_activation_counts = self.layer.total_MAC_count / self.mapping.spatial_mapping.unit_count["W"][0]
            # calc number of active columns
            mapped_act_dim = tuple(self.mapping.spatial_mapping.mapping_dict_origin[act_layer_op][0])
            if len(mapped_act_dim) == 0:
                active_cols = 1
            elif self.is_nested_tuple(mapped_act_dim):
                active_cols = 1
                for ele in mapped_act_dim:
                    active_cols *= ele[1]
            else:
                active_cols = mapped_act_dim[0][1]
            # calc number of active rows
            mapped_output_dim = tuple(self.mapping.spatial_mapping.mapping_dict_origin[output_layer_op][0])
            if len(mapped_output_dim) == 0:
                active_rows = 1
            elif self.is_nested_tuple(mapped_output_dim):
                active_rows = 1
                for ele in mapped_output_dim:
                    active_rows *= ele[1]
            else:
                active_rows = mapped_output_dim[0][1]
            # calc number of active macros
            active_macros = self.mapping.spatial_mapping.unit_count["W"][0] / (active_cols * active_rows)
            # calc total number of adder inputs (along the column)
            for mem_level in self.accelerator.cores[0].memory_hierarchy.mem_level_list:
                if mem_level.operands[0] == output_mem_op:
                    lowest_output_mem_level = mem_level
                    break
            nb_inputs_of_adder = sorted(lowest_output_mem_level.served_dimensions)[0].size
            # get adder input precision
            adder_input_pres = self.layer.operand_precision[output_layer_op]  # partial sum precision
            # get 1b adder energy
            nd2_cap = 0.7 / 1e3  # unit: pF
            xor2_cap = 0.7 * 1.5 / 1e3  # unit: pF
            vdd = 0.9  # unit: V
            adder_energy_fa = (3 * nd2_cap + 2 * xor2_cap) * (vdd ** 2)  # unit: pJ
            adder_energy_ha = (2 * xor2_cap) * (vdd ** 2)  # unit: pJ
            # calc energy
            adder_tree_energy = self.get_adder_trees_energy(adder_input_pres=adder_input_pres,
                                                            nb_inputs_of_adder=nb_inputs_of_adder,
                                                            mapped_inputs=active_rows,
                                                            adder_energy_fa=adder_energy_fa,
                                                            adder_energy_ha=adder_energy_ha,
                                                            mapped_cols=active_cols,
                                                            mapped_macros=active_macros,
                                                            macro_activation_counts=macro_activation_counts)
        return adder_tree_energy

    def get_adder_trees_energy(self, adder_input_pres, nb_inputs_of_adder, mapped_inputs, adder_energy_fa,
                               adder_energy_ha, mapped_cols, mapped_macros, macro_activation_counts):
        """
        get the energy spent on RCA adder trees for specific layer and mapping
        """
        adder_depth = log2(nb_inputs_of_adder)
        assert nb_inputs_of_adder % 1 == 0, \
            f"The number of inputs for an adder tree [{nb_inputs_of_adder}] is not in the power of 2."
        adder_depth = int(adder_depth)  # float -> int for simplicity
        adder_output_pres = adder_input_pres + adder_depth
        nb_of_1b_adder = nb_inputs_of_adder * (adder_input_pres + 1) - (
                    adder_input_pres + adder_depth + 1)  # nb of 1b adders in a single adder tree

        # In the adders' model, we classify the basic FA (1-b full adder) as two types:
        # 1. fully activated FA: two of its inputs having data comes in. (higher energy cost)
        # 2. half activated FA: only one of its inputs having data comes in.
        # The 2nd type has lower energy cost, because no carry will be generated and the carry path stays unchanged.
        # Below we figure out how many there are of fully activated FA and half activated FA
        if mapped_inputs >= 1:
            if mapped_inputs >= nb_inputs_of_adder:
                """
                :param fully_activated_number_of_1b_adder: fully activated 1b adder, probably will produce a carry
                :param half_activated_number_of_1b_adder: only 1 input is activate and the other port is 0, so carry path is activated.
                """
                fully_activated_number_of_1b_adder = nb_of_1b_adder
                half_activated_number_of_1b_adder = 0
            else:
                """
                find out fully_activated_number_of_1b_adder and half_activated_number_of_1b_adder when inputs are not fully mapped.
                method: iteratively check if left_input is bigger or smaller than baseline, which will /2 each time, until left_input == 1
                :param left_input: the number of inputs waiting for processing
                :param baseline: serves as references for left_input
                """
                fully_activated_number_of_1b_adder = 0
                half_activated_number_of_1b_adder = 0
                left_input = mapped_inputs
                baseline = nb_inputs_of_adder
                while left_input != 0:
                    baseline = baseline / 2
                    activated_depth = int(log2(baseline))
                    if left_input <= 1 and baseline == 1:  # special case
                        fully_activated_number_of_1b_adder += 0
                        half_activated_number_of_1b_adder += adder_input_pres
                        left_input = 0
                    elif left_input > baseline:
                        fully_activated_number_of_1b_adder += baseline * (adder_input_pres + 1) - (
                                    adder_input_pres + activated_depth + 1) + (adder_input_pres + activated_depth)
                        half_activated_number_of_1b_adder += 0
                        left_input = left_input - baseline
                    elif left_input < baseline:
                        half_activated_number_of_1b_adder += adder_input_pres + activated_depth
                    else:  # left_input == baseline
                        fully_activated_number_of_1b_adder += baseline * (adder_input_pres + 1) - (
                                    adder_input_pres + activated_depth + 1)
                        half_activated_number_of_1b_adder += adder_input_pres + activated_depth
                        left_input = left_input - baseline

            single_adder_tree_energy = fully_activated_number_of_1b_adder * adder_energy_fa + \
                                       half_activated_number_of_1b_adder * adder_energy_ha
            nb_of_activation_times = mapped_cols * mapped_macros * macro_activation_counts
            energy_adders = single_adder_tree_energy * nb_of_activation_times
        else:
            energy_adders = 0
        return energy_adders

    ## Computes the memories reading/writing energy by converting the access patterns in self.mapping to
    # energy breakdown using the memory hierarchy of the core on which the layer is mapped.
    #
    # The energy breakdown is saved in self.energy_breakdown.
    #
    # The energy total consumption is saved in self.energy_total.
    def calc_memory_energy_cost(self):
        core = self.accelerator.get_core(self.core_id)
        mem_hierarchy = core.memory_hierarchy

        energy_breakdown = {}
        energy_breakdown_further = {}
        energy_total = 0
        for (layer_op, mem_access_list_per_op) in self.memory_word_access.items():
            """Retrieve the memory levels in the hierarchy for this memory operand"""
            mem_op = self.layer_op_to_mem_op[layer_op]
            memory_levels = mem_hierarchy.get_memory_levels(mem_op=mem_op)

            breakdown = (
                []
            )  # Stores the energy breakdown of a single layer operand (W, I, ...)
            breakdown_further = []  # Stores
            for (access_count, memory_level) in zip(
                    mem_access_list_per_op, memory_levels
            ):
                energy_cost_per_read_out = memory_level.read_energy
                energy_cost_per_write_in = memory_level.write_energy
                read_out_energy_to_above = access_count.get_total_read_outs_to_above(
                    scaling=energy_cost_per_read_out
                )
                write_in_energy_from_above = (
                    access_count.get_total_write_ins_from_above(
                        scaling=energy_cost_per_write_in
                    )
                )
                read_out_energy_to_below = access_count.get_total_read_outs_to_below(
                    scaling=energy_cost_per_read_out
                )
                write_in_energy_from_below = (
                    access_count.get_total_write_ins_from_below(
                        scaling=energy_cost_per_write_in
                    )
                )
                total_read_out_energy = (
                        read_out_energy_to_above + read_out_energy_to_below
                )
                total_write_in_energy = (
                        write_in_energy_from_above + write_in_energy_from_below
                )
                total_energy_cost_memory = total_read_out_energy + total_write_in_energy
                breakdown.append(
                    total_energy_cost_memory
                )  # Here the breakdown only saves the total energy cost per memory level
                breakdown_further.append(
                    FourWayDataMoving(
                        read_out_energy_to_below,
                        write_in_energy_from_below,
                        read_out_energy_to_above,
                        write_in_energy_from_above,
                    )
                )  # here it contains the full split
                energy_total += total_energy_cost_memory
            energy_breakdown[layer_op] = breakdown
            energy_breakdown_further[layer_op] = breakdown_further
        self.energy_breakdown = energy_breakdown
        self.energy_breakdown_further = energy_breakdown_further
        self.mem_energy = energy_total
        self.energy_total = self.mem_energy + self.MAC_energy
        logger.debug(f"Ran {self}. Total energy = {self.energy_total}")

    ##  Calculate latency in 4 steps
    #
    # 1) As we already calculated the ideal data transfer rate in combined_mapping.py (in the Mapping class),
    # here we start with calculating the required (or allowed) memory updating window by comparing the effective
    # data size with the physical memory size at each level. If the effective data size is smaller than 50%
    # of the physical memory size, then we take the whole period as the allowed memory updating window (double buffer effect);
    # otherwise we take the the period divided by the top_ir_loop as the allowed memory updating window.
    #
    # 2) Then, we compute the real data transfer rate given the actual memory bw per functional port pair,
    # assuming we have enough memory ports.
    #
    # 3) In reality, there is no infinite memory port to use. So, as the second step, we combine the real
    # data transfer attributes per physical memory port.
    #
    # 4) Finally, we combine the stall/slack of each memory port to get the final latency.
    def calc_latency(self):
        self.calc_double_buffer_flag()
        self.calc_allowed_and_real_data_transfer_cycle_per_DTL()
        self.combine_data_transfer_rate_per_physical_port()
        self.calc_data_loading_offloading_latency()
        self.calc_overall_latency()

    ## This function checks the double-buffer possibility for each operand at each memory level
    # (minimal memory BW requirement case) by comparing the physical memory size with the effective
    # data size, taking into account the memory sharing between operands.
    def calc_double_buffer_flag(self):
        double_buffer_true = {}
        output_layer_op = self.layer.output_operand
        for layer_op in self.layer.operand_list:
            mem_op = self.layer_op_to_mem_op[layer_op]
            """ start with False for each operand at the lowest arch level (MAC array level) """
            double_buffer_true[layer_op] = [False]
            for mem_lv in range(0, self.mapping_int.mem_level[layer_op]):
                if self.effective_mem_utili_shared[layer_op][mem_lv] <= 0.5 and (layer_op != output_layer_op):
                    double_buffer_true[layer_op].append(True)
                elif (
                        self.effective_mem_utili_individual[layer_op][mem_lv]
                        <= 1 - self.effective_mem_utili_shared[layer_op][mem_lv]
                ) and (layer_op != output_layer_op):
                    double_buffer_true[layer_op].append(True)
                    shared_mem_list = get_shared_mem_list(
                        mem_op, mem_lv, self.mem_sharing_list
                    )
                    """ When one of the operand in the shared memory get the "double-buffer" chance, 
                    all operands of that shared memory level need to update the memory utilization 
                    for later memory free space evaluation """
                    for shared_mem_op, shared_mem_lv in shared_mem_list:
                        try:
                            shared_layer_op = self.mem_op_to_layer_op[shared_mem_op]
                        except:  # mem op to layer op might not have this mem op (e.g. pooling layer)
                            continue
                        self.effective_mem_utili_shared[shared_layer_op][
                            shared_mem_lv
                        ] += self.effective_mem_utili_individual[layer_op][mem_lv]
                else:
                    double_buffer_true[layer_op].append(False)

        self.double_buffer_true = double_buffer_true

    ## Construct a 4-way data transfer pattern for each unit mem, calculate
    # {allowed_mem_updating_cycle, real_data_trans_cycle, DTL_SS_cycle} per period
    def calc_allowed_and_real_data_transfer_cycle_per_DTL(self):
        allowed_mem_updat_cycle = {}
        real_data_trans_cycle = {}
        """ stall (+) or slack (-) cycle within each period per virtual data transfer link (DTL) """
        DTL_SS_cycle = {}

        for layer_op in self.layer.operand_list:
            allowed_mem_updat_cycle[layer_op] = []
            real_data_trans_cycle[layer_op] = []
            DTL_SS_cycle[layer_op] = []
            mem_op = self.layer_op_to_mem_op[layer_op]
            for mem_lv in range(self.mapping_int.mem_level[layer_op]):
                """======================================allowed_mem_updating_cycle(below)====================================="""
                """ wr_in_by_low & rd_out_to_low"""
                if self.double_buffer_true[layer_op][mem_lv]:
                    wr_in_by_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.wr_in_by_low
                    rd_out_to_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.rd_out_to_low
                else:
                    wr_in_by_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.wr_in_by_low
                    rd_out_to_low_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.rd_out_to_low

                """ wr_in_by_high & rd_out_to_high """
                if self.double_buffer_true[layer_op][mem_lv + 1]:
                    wr_in_by_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.wr_in_by_high
                    rd_out_to_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].data_trans_period.rd_out_to_high
                else:
                    wr_in_by_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.wr_in_by_high
                    rd_out_to_high_allowed = self.mapping_int.unit_mem_data_movement[
                        layer_op
                    ][mem_lv].inst_data_trans_window.rd_out_to_high

                """ All """
                updating_window = FourWayDataMoving(
                    rd_out_to_low_allowed,
                    wr_in_by_low_allowed,
                    rd_out_to_high_allowed,
                    wr_in_by_high_allowed,
                )
                allowed_mem_updat_cycle[layer_op].append(updating_window)
                """ ======================================allowed_mem_updating_cycle(above)===================================== """

                """ =========================================real_data_trans_cycle(below)======================================== """
                """ wr_in_by_low """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.wr_in_by_low
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.wr_in_by_low
                mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                wr_in_by_low_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ rd_out_to_low """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.rd_out_to_low
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.rd_out_to_low
                mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                rd_out_to_low_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ rd_out_to_high """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.rd_out_to_high
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.rd_out_to_high
                mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                rd_out_to_high_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ wr_in_by_high """
                data_precision = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_precision.wr_in_by_high
                data_trans_amount = self.mapping_int.unit_mem_data_movement[layer_op][
                    mem_lv
                ].data_trans_amount_per_period.wr_in_by_high
                mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                wr_in_by_high_real = ceil(data_trans_amount * data_precision / mem_bw)

                """ All """
                real_data_trans = FourWayDataMoving(
                    rd_out_to_low_real,
                    wr_in_by_low_real,
                    rd_out_to_high_real,
                    wr_in_by_high_real,
                )
                real_data_trans_cycle[layer_op].append(real_data_trans)
                """ =========================================real_data_trans_cycle(above)======================================= """

        self.allowed_mem_updat_cycle = allowed_mem_updat_cycle
        self.real_data_trans_cycle = real_data_trans_cycle

    ## Consider memory sharing and port sharing, combine the data transfer activity
    # Step 1: collect port activity per memory instance per physical memory port
    # Step 2: calculate SS combine and MUW union parameters per physical memory port
    def combine_data_transfer_rate_per_physical_port(self):
        # Step 1: collect port activity per memory instance per physical memory port
        port_activity_collect = []
        for mem_instance in self.mem_level_list:
            port_activity_single = {}
            port_list = mem_instance.port_list
            for port in port_list:
                port_activity_single[str(port)] = []
                for mem_op, mem_lv, mov_dir in port.served_op_lv_dir:
                    try:
                        layer_op = self.mem_op_to_layer_op[mem_op]
                    except:  # mem op to layer might not have this mem op (e.g. pooling layer)
                        continue
                    period_count = getattr(
                        self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period_count,
                        mov_dir,
                    )
                    if period_count == 0:
                        # skip the inactive data movement activities because they won't impact SS
                        continue
                    period = getattr(
                        self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period,
                        mov_dir,
                    )
                    real_cycle = getattr(
                        self.real_data_trans_cycle[layer_op][mem_lv], mov_dir
                    )
                    allowed_cycle = getattr(
                        self.allowed_mem_updat_cycle[layer_op][mem_lv], mov_dir
                    )
                    port_activity = PortActivity(
                        real_cycle,
                        allowed_cycle,
                        period,
                        period_count,
                        layer_op,
                        mem_lv,
                        mov_dir,
                    )
                    port_activity_single[str(port)].append(port_activity)
            port_activity_collect.append(port_activity_single)
        self.port_activity_collect = port_activity_collect

        # Step 2: calculate SS combine and MUW union parameters per physical memory port
        SS_comb_collect = [
            {port: None for port in mem_ports} for mem_ports in port_activity_collect
        ]
        SS_comb_list = [0]
        # intermediate parameters saved for debugging purpose
        MUW_union_collect = [
            {port: None for port in mem_ports} for mem_ports in port_activity_collect
        ]

        for idx, mem_ports in enumerate(port_activity_collect):
            for port_name, port_activity in mem_ports.items():
                if len(port_activity) == 1:
                    MUW_union_collect[idx][port_name] = port_activity[0].allowed_cycle
                    SS_comb_collect[idx][port_name] = port_activity[0].SS
                    SS_comb_list.append(port_activity[0].SS)
                elif len(port_activity) != 0:
                    MUW_union_collect[idx][port_name] = calc_MUW_union(port_activity)
                    SS_positive_sum = 0
                    SS_negative_sum = 0
                    MUW_sum = 0
                    for port_d in port_activity:
                        if port_d.SS > 0:
                            SS_positive_sum += port_d.SS
                        else:
                            SS_negative_sum += port_d.SS
                        MUW_sum += port_d.MUW
                    SS_comb = SS_positive_sum + max(
                        0, SS_negative_sum + MUW_sum - MUW_union_collect[idx][port_name]
                    )
                    SS_comb_collect[idx][port_name] = SS_comb
                    SS_comb_list.append(SS_comb)

        self.MUW_union_collect = MUW_union_collect
        self.SS_comb_collect = SS_comb_collect
        # Assuming all the memory ports can work in parallel
        self.SS_comb = max(SS_comb_list)

    ## Calculate the initial/final data loading/off-loading cycle by separating out
    # the first-time input operands' / the last-time output operand's data movement
    # on corresponding ports.
    def calc_data_loading_offloading_latency(self):
        # Collect ports' initial data-loading and final data-offloading activities
        data_loading_per_mem_inst = []
        data_loading_cc_per_op = {op: {} for op in self.layer.input_operands}
        data_offloading_per_mem_inst = []
        data_offloading_cc_per_op = {}
        for mem_inst_idx, mem_instance in enumerate(self.mem_level_list):
            data_loading_single = {}
            data_offloading_single = {}
            port_list = mem_instance.port_list
            for port in port_list:
                data_loading_single[str(port)] = []
                data_offloading_single[str(port)] = []
                served_operands = set(
                    s[0] for s in port.served_op_lv_dir if s[0] in ["I1", "I2"]
                )
                port_is_shared_by_two_input_operands = len(served_operands) > 1
                for mem_op, mem_lv, mov_dir in port.served_op_lv_dir:
                    try:
                        layer_op = self.mem_op_to_layer_op[mem_op]
                    except:  # mem op to layer op might not have this mem op (e.g. pooling layer)
                        continue
                    period_count = getattr(
                        self.mapping_int.unit_mem_data_movement[layer_op][
                            mem_lv
                        ].data_trans_period_count,
                        mov_dir,
                    )
                    if period_count == 0:
                        # skip for the inactive data movement
                        continue
                    if mem_op in ["I1", "I2"]:
                        real_cycle = getattr(
                            self.real_data_trans_cycle[layer_op][mem_lv], mov_dir
                        )
                        data_in_charge = getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_trans_amount_per_period,
                            mov_dir,
                        ) * getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_precision,
                            mov_dir,
                        )
                        if mov_dir[:2] == "rd":
                            mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                        else:
                            mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                        port_activity = PortBeginOrEndActivity(
                            real_cycle,
                            data_in_charge,
                            mem_bw,
                            layer_op,
                            mem_lv,
                            mov_dir,
                        )
                        data_loading_single[str(port)].append(port_activity)
                        data_loading_cc_per_op[layer_op][
                            layer_op + str(mem_lv) + "_" + mov_dir
                            ] = (real_cycle, port_is_shared_by_two_input_operands)
                    else:
                        if mov_dir in ["rd_out_to_low", "wr_in_by_high"]:
                            # don't consider partial sum flowing in the final data off-loading stage
                            continue
                        real_cycle = getattr(
                            self.real_data_trans_cycle[layer_op][mem_lv], mov_dir
                        )
                        data_in_charge = getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_trans_amount_per_period,
                            mov_dir,
                        ) * getattr(
                            self.mapping_int.unit_mem_data_movement[layer_op][
                                mem_lv
                            ].data_precision,
                            mov_dir,
                        )
                        if mov_dir[:2] == "rd":
                            mem_bw = self.mem_r_bw_dict[mem_op][mem_lv]
                        else:
                            mem_bw = self.mem_w_bw_dict[mem_op][mem_lv]
                        port_activity = PortBeginOrEndActivity(
                            real_cycle,
                            data_in_charge,
                            mem_bw,
                            layer_op,
                            mem_lv,
                            mov_dir,
                        )
                        data_offloading_single[str(port)].append(port_activity)
                        data_offloading_cc_per_op[
                            layer_op + str(mem_lv) + "_" + mov_dir
                            ] = real_cycle

            data_loading_per_mem_inst.append(data_loading_single)
            data_offloading_per_mem_inst.append(data_offloading_single)
        self.data_loading_per_mem_inst = data_loading_per_mem_inst
        self.data_loading_cc_per_op = data_loading_cc_per_op
        self.data_offloading_per_mem_inst = data_offloading_per_mem_inst
        self.data_offloading_per_op = data_offloading_cc_per_op

        # Combine ports' initial data-loading activities to get the data loading cycle amount
        data_loading_cc_pair_combined_per_op = {
            op: [] for op in self.layer.input_operands
        }
        data_loading_individual_part = {op: 0 for op in self.layer.input_operands}
        data_loading_half_shared_part = {op: 0 for op in self.layer.input_operands}
        data_loading_shared_part = {op: 0 for op in self.layer.input_operands}
        for layer_op in self.layer.input_operands:
            for mem_lv in range(self.active_mem_level[layer_op] - 1):
                elem1 = data_loading_cc_per_op[layer_op][
                    layer_op + str(mem_lv) + "_" + "wr_in_by_high"
                    ]
                elem2 = data_loading_cc_per_op[layer_op][
                    layer_op + str(mem_lv + 1) + "_" + "rd_out_to_low"
                    ]
                completely_shared = elem1[1] and elem2[1]
                completely_separate = not (elem1[1]) and not (elem2[1])
                longest_loading_cc = max(elem1[0], elem2[0])
                # for the ports that serve the same data movement purpose, take the longest data loading cycle
                data_loading_cc_pair_combined = longest_loading_cc
                data_loading_cc_pair_combined_per_op[layer_op].append(
                    data_loading_cc_pair_combined
                )
                if completely_separate:
                    data_loading_individual_part[layer_op] += longest_loading_cc
                elif completely_shared:
                    data_loading_shared_part[layer_op] += longest_loading_cc
                else:
                    # the data transfer link between two memory levels is half-shared,
                    # i.e. on one memory side, the port is shared, while on another memory side,
                    # there are different memories with separate ports
                    data_loading_half_shared_part[layer_op] = longest_loading_cc

        if len(self.layer.input_operands) == 1:
            data_loading_cycle = data_loading_individual_part[
                self.layer.input_operands[0]
            ]
        else:
            op1 = self.layer.input_operands[0]
            op2 = self.layer.input_operands[1]
            possible1 = data_loading_shared_part[op1] + max(
                data_loading_shared_part[op2]
                + data_loading_half_shared_part[op2]
                + data_loading_individual_part[op2],
                data_loading_half_shared_part[op1] + data_loading_individual_part[op1],
            )
            possible2 = data_loading_shared_part[op2] + max(
                data_loading_shared_part[op1]
                + data_loading_half_shared_part[op1]
                + data_loading_individual_part[op1],
                data_loading_half_shared_part[op2] + data_loading_individual_part[op2],
            )
            data_loading_cycle = min(possible1, possible2)

        self.data_loading_cc_pair_combined_per_op = data_loading_cc_pair_combined_per_op
        self.data_loading_individual_part = data_loading_individual_part
        self.data_loading_half_shared_part = data_loading_half_shared_part
        self.data_loading_shared_part = data_loading_shared_part
        self.data_loading_cycle = data_loading_cycle

        # Combine ports' final data-offloading activities to get the data offloading cycle amount
        # TODO Only considered the worst case for now
        #  (assumed that all the ports are working in series during the final data off-loading phase)
        data_offloading_cc_pair_combined = []
        layer_op = self.layer.output_operand
        for mem_lv in range(self.active_mem_level[layer_op] - 1):
            elem1 = data_offloading_cc_per_op[
                layer_op + str(mem_lv) + "_" + "rd_out_to_high"
                ]
            elem2 = data_offloading_cc_per_op[
                layer_op + str(mem_lv + 1) + "_" + "wr_in_by_low"
                ]
            longest_offloading_cc = max(elem1, elem2)
            # for the ports that serve the same data movement purpose, take the longest data loading cycle
            data_offloading_cc_pair_combined.append(longest_offloading_cc)
        data_offloading_cycle = sum(data_offloading_cc_pair_combined)

        self.data_offloading_cc_pair_combined = data_offloading_cc_pair_combined
        self.data_offloading_cycle = data_offloading_cycle

    ## This function integrates the previous calculated SScomb, data loading and off-loading cycle to get the overall latency
    def calc_overall_latency(self):
        # the ideal cycle count assuming the MAC array is 100% utilized
        ideal_cycle = ceil(
            self.layer.total_MAC_count
            / self.accelerator.get_core(self.core_id).operational_array.total_unit_count
        )

        # the ideal temporal cycle count given the spatial mapping (the spatial mapping can be non-ideal)
        ideal_temporal_cycle = self.mapping_int.temporal_mapping.total_cycle
        MAC_spatial_utilization = ideal_cycle / ideal_temporal_cycle

        # Total latency without the initial data loading and the final data off-loading
        latency_total0 = ideal_temporal_cycle + self.SS_comb
        MAC_utilization0 = ideal_cycle / latency_total0

        # Total latency with the initial data loading, but without the final data off-loading
        latency_total1 = ideal_temporal_cycle + self.SS_comb + self.data_loading_cycle
        MAC_utilization1 = ideal_cycle / latency_total1

        # Total latency with both the initial data loading and the final data off-loading
        latency_total2 = (
                ideal_temporal_cycle
                + self.SS_comb
                + self.data_loading_cycle
                + self.data_offloading_cycle
        )
        MAC_utilization2 = ideal_cycle / latency_total2

        self.ideal_cycle = ideal_cycle
        self.ideal_temporal_cycle = ideal_temporal_cycle
        self.MAC_spatial_utilization = MAC_spatial_utilization
        self.latency_total0 = latency_total0
        self.latency_total1 = latency_total1
        self.latency_total2 = latency_total2
        self.MAC_utilization0 = MAC_utilization0
        self.MAC_utilization1 = MAC_utilization1
        self.MAC_utilization2 = MAC_utilization2

    def __add__(self, other):
        sum = pickle_deepcopy(self)

        ## Energy
        sum.MAC_energy += other.MAC_energy
        sum.mem_energy += other.mem_energy
        for op in sum.energy_breakdown.keys():
            if op in other.energy_breakdown.keys():
                l = []
                for i in range(
                        min(len(self.energy_breakdown[op]), len(other.energy_breakdown[op]))
                ):
                    l.append(
                        self.energy_breakdown[op][i] + other.energy_breakdown[op][i]
                    )
                i = min(len(self.energy_breakdown[op]), len(other.energy_breakdown[op]))
                l += self.energy_breakdown[op][i:]
                l += other.energy_breakdown[op][i:]
                sum.energy_breakdown[op] = l

        for op in sum.energy_breakdown_further.keys():
            if op in other.energy_breakdown_further.keys():
                l = []
                for i in range(
                        min(
                            len(self.energy_breakdown_further[op]),
                            len(other.energy_breakdown_further[op]),
                        )
                ):
                    l.append(
                        self.energy_breakdown_further[op][i]
                        + other.energy_breakdown_further[op][i]
                    )
                i = min(
                    len(self.energy_breakdown_further[op]),
                    len(other.energy_breakdown_further[op]),
                )
                l += self.energy_breakdown_further[op][i:]
                l += other.energy_breakdown_further[op][i:]
                sum.energy_breakdown_further[op] = l

        # Get all the operands from other that are not in self and add them to the energy breakdown aswell
        op_diff = set(other.energy_breakdown.keys()) - set(self.energy_breakdown.keys())
        for op in op_diff:
            sum.energy_breakdown[op] = other.energy_breakdown[op]
            sum.energy_breakdown_further[op] = other.energy_breakdown_further[op]

        sum.energy_total += other.energy_total

        ## Memory access
        for op in sum.memory_word_access.keys():
            if op in other.memory_word_access.keys():
                l = []
                for i in range(
                        min(
                            len(self.memory_word_access[op]),
                            len(other.memory_word_access[op]),
                        )
                ):
                    l.append(
                        self.memory_word_access[op][i] + other.memory_word_access[op][i]
                    )
                i = min(
                    len(self.memory_word_access[op]), len(other.memory_word_access[op])
                )
                l += self.memory_word_access[op][i:]
                l += other.memory_word_access[op][i:]
                sum.memory_word_access[op] = l
        for op in op_diff:
            sum.memory_word_access[op] = other.memory_word_access[op]

        ## Latency
        sum.data_loading_cycle += other.data_loading_cycle
        sum.data_offloading_cycle += other.data_offloading_cycle
        sum.ideal_cycle += other.ideal_cycle
        sum.ideal_temporal_cycle += other.ideal_temporal_cycle
        sum.SS_comb += other.SS_comb
        sum.latency_total0 += other.latency_total0
        sum.latency_total1 += other.latency_total1
        sum.latency_total2 += other.latency_total2

        ## MAC utilization
        sum.MAC_spatial_utilization = sum.ideal_cycle / sum.ideal_temporal_cycle
        sum.MAC_utilization0 = sum.ideal_cycle / sum.latency_total0
        sum.MAC_utilization1 = sum.ideal_cycle / sum.latency_total1
        sum.MAC_utilization2 = sum.ideal_cycle / sum.latency_total2

        ## layer
        if type(sum.layer) != list:
            sum.layer = [sum.layer.id]
        if type(other.layer) != list:
            other_layer = [other.layer.id]
        sum.layer += other_layer

        ## core_id
        if type(sum.core_id) != list:
            sum.core_id = [sum.core_id]
        if type(other.layer) != list:
            other_core_id = [other.core_id]
        sum.core_id += other_core_id

        ## Not addable
        func = [
            "calc_allowed_and_real_data_transfer_cycle_per_DTL",
            "calc_data_loading_offloading_latency",
            "calc_double_buffer_flag",
            "calc_overall_latency",
            "calc_MAC_energy_cost",
            "calc_energy",
            "calc_latency",
            "calc_memory_energy_cost",
            "calc_memory_utilization",
            "calc_memory_word_access",
            "combine_data_transfer_rate_per_physical_port",
            "run",
            "calc_adders_energy",
            "get_adder_trees_energy",
            "is_nested_tuple",
        ]
        add_attr = [
            "MAC_energy",
            "mem_energy",
            "energy_breakdown",
            "energy_breakdown_further",
            "energy_total",
            "memory_word_access",
            "data_loading_cycle",
            "data_offloading_cycle",
            "ideal_cycle",
            "ideal_temporal_cycle",
            "SS_comb",
            "latency_total0",
            "latency_total1",
            "latency_total2",
            "MAC_spatial_utilization",
            "MAC_utilization0",
            "MAC_utilization1",
            "MAC_utilization2",
            "layer",
            "core_id",
        ]

        if hasattr(self, "accelerator") and hasattr(other, "accelerator"):
            if self.accelerator.name.startswith(other.accelerator.name):
                sum.accelerator = other.accelerator
                add_attr.append("accelerator")
            elif other.accelerator.name.startswith(self.accelerator.name):
                add_attr.append("accelerator")
        else:
            pass

        for attr in dir(sum):
            if attr not in (func + add_attr) and attr[0] != "_":
                delattr(sum, attr)

        return sum

    def __mul__(self, number):
        mul = pickle_deepcopy(self)

        # Energy
        mul.MAC_energy *= number
        mul.mem_energy *= number
        mul.energy_breakdown = {
            op: [
                mul.energy_breakdown[op][i] * number
                for i in range(len(mul.energy_breakdown[op]))
            ]
            for op in mul.energy_breakdown.keys()
        }
        mul.energy_breakdown_further = {
            op: [
                mul.energy_breakdown_further[op][i] * number
                for i in range(len(mul.energy_breakdown_further[op]))
            ]
            for op in mul.energy_breakdown_further.keys()
        }
        mul.energy_total *= number

        # Memory access
        mul.memory_word_access = {
            op: [
                mul.memory_word_access[op][i] * number
                for i in range(len(mul.memory_word_access[op]))
            ]
            for op in mul.memory_word_access.keys()
        }

        # Latency
        mul.data_loading_cycle *= number
        mul.data_offloading_cycle *= number
        mul.ideal_cycle *= number
        mul.ideal_temporal_cycle *= number
        mul.latency_total0 *= number
        mul.latency_total1 *= number
        mul.latency_total2 *= number

        # MAC utilization
        mul.MAC_spatial_utilization = mul.ideal_cycle / mul.ideal_temporal_cycle
        mul.MAC_utilization0 = mul.ideal_cycle / mul.latency_total0
        mul.MAC_utilization1 = mul.ideal_cycle / mul.latency_total1
        mul.MAC_utilization2 = mul.ideal_cycle / mul.latency_total2

        # Not addable
        func = [
            "calc_allowed_and_real_data_transfer_cycle_per_DTL",
            "calc_data_loading_offloading_latency",
            "calc_double_buffer_flag",
            "calc_overall_latency",
            "calc_MAC_energy_cost",
            "calc_energy",
            "calc_latency",
            "calc_memory_energy_cost",
            "calc_memory_utilization",
            "calc_memory_word_access",
            "combine_data_transfer_rate_per_physical_port",
            "run",
            "calc_adders_energy",
            "get_adder_trees_energy",
            "is_nested_tuple",
        ]
        mul_attr = [
            "MAC_energy",
            "mem_energy",
            "energy_breakdown",
            "energy_breakdown_further",
            "energy_total",
            "memory_word_access",
            "data_loading_cycle",
            "data_offloading_cycle",
            "ideal_cycle",
            "ideal_temporal_cycle",
            "latency_total0",
            "latency_total1",
            "latency_total2",
            "MAC_spatial_utilization",
            "MAC_utilization0",
            "MAC_utilization1",
            "MAC_utilization2",
            "layer",
            "accelerator",
        ]

        for attr in dir(mul):
            if attr not in (func + mul_attr) and attr[0] != "_":
                delattr(mul, attr)

        return mul

    @staticmethod
    def is_nested_tuple(obj):
        if isinstance(obj, tuple):
            for item in obj:
                if isinstance(item, tuple):
                    # If any item within the tuple is itself a tuple, it's a nested tuple
                    return True
        return False
