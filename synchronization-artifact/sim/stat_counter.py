import numpy as np
from pyparsing import delimitedList
import pickle

# Take an input Stim circuit and derive:
# - Total execution time 
# - Max concurrent operations at every time step

class stat_counter():
    def __init__(self, gate_profile, coord_to_index_mapper, 
                 gates, gates1Q, gates2Q, measures) -> None:
        # Receive Stim circuit from circuit class
        self.timer = {}
        self.frames = {}
        self.mapper = coord_to_index_mapper
        self.gate_profile = gate_profile
        self.__gates1Q = gates1Q
        self.__gates2Q = gates2Q
        self.__measures = measures
        self.__gates = gates
        self.ops = {'CX':2, '1Q':1, 'M':1}
        pass

    def parse_circuit(self, ckt:str):
        frames = {}
        frame = 0
        for line in ckt.split('\n'):
            if 'TICK' in line:
                # Marks the start of a frame.
                frame += 1
                frames[frame] = []
                pass
            else:
                # Circuit pragmas
                # Every instruction after a tick can be a gate or a circuit annotation
                inst = delimitedList(line, delim=' ')
                gate = inst[0]
                qubit = inst[1] # For 2Q gates, the latency is determined by the control
                if gate not in self.__gates:
                    continue # circuit annotation
                qubit = self.mapper[int(qubit)] # int to coord
                if gate in self.__gates1Q:
                    type = '1Q'
                elif gate in self.__measures:
                    type = 'M'
                else:
                    type = 'CX'
                if qubit not in self.timer.keys():
                    self.timer[qubit] = self.gate_profile[qubit]['latency'][type]
                else:
                    self.timer[qubit] += self.gate_profile[qubit]['latency'][type]
                frames[frame].append(self.ops[type]) # Append this gate to all the gates executed in this frame
                pass
            pass
        self.frames = frames
        return
    
    def stats(self):
        # Print to file and dump pickled file too
        return
    pass