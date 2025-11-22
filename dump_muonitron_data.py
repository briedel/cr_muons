from icecube import icetray, dataclasses, dataio, phys_services, interfaces
from icecube.icetray import I3Tray, I3Units
from os.path import expandvars
from icecube.MuonGun import MuonPropagator, Crust, Cylinder #, Sphere
from icecube.phys_services import Sphere

import numpy as np
from collections import defaultdict

import argparse
import random
import math
import os
import json
import h5py

OUTPUT_FILE = "training_data.h5"

def write_to_hdf5(input_iterator):
    # 1. Open file
    with h5py.File(OUTPUT_FILE, "w") as f:

        # 2. Create Resizable Datasets
        # We don't know total size yet, so we make them resizable (maxshape=None)
        dset_cond = f.create_dataset("conditions", shape=(0, 4), maxshape=(None, 4), dtype='f4', chunks=(10000, 4))
        dset_muons = f.create_dataset("muons", shape=(0, 3), maxshape=(None, 3), dtype='f4', chunks=(100000, 3))
        dset_counts = f.create_dataset("counts", shape=(0,), maxshape=(None,), dtype='i4', chunks=(10000,))

        # Buffers
        buf_cond, buf_muons, buf_counts = [], [], []
        CHUNK_SIZE = 50000

        for event in input_iterator:
            # ... (Logic to extract cond, muons, count from event) ...
            # Example mock data:
            cond = [1.0, 0.5, 1.0, 2.5]
            muons = [[10.0, 5.0, -5.0], [12.0, 4.0, -4.0]]
            count = 2

            buf_cond.append(cond)
            buf_muons.extend(muons)
            buf_counts.append(count)

            # 3. Flush when buffer is full
            if len(buf_cond) >= CHUNK_SIZE:
                _append_to_dsets(dset_cond, dset_muons, dset_counts, buf_cond, buf_muons, buf_counts)
                buf_cond, buf_muons, buf_counts = [], [], [] # Clear

        # 4. Final Flush
        if buf_cond:
            _append_to_dsets(dset_cond, dset_muons, dset_counts, buf_cond, buf_muons, buf_counts)

def _append_to_dsets(ds_c, ds_m, ds_n, b_c, b_m, b_n):
    # Current size
    curr_c = ds_c.shape[0]
    curr_m = ds_m.shape[0]
    curr_n = ds_n.shape[0]

    # New size
    new_c = curr_c + len(b_c)
    new_m = curr_m + len(b_m)
    new_n = curr_n + len(b_n)

    # Resize
    ds_c.resize((new_c, 4))
    ds_m.resize((new_m, 3))
    ds_n.resize((new_n,))

    # Write
    ds_c[curr_c:] = b_c
    ds_m[curr_m:] = b_m
    ds_n[curr_n:] = b_n
    print(f"Wrote batch. Total events: {new_n}")

class convert_muonitron_jsonl(icetray.I3ConditionalModule):
    """
    Class to aggregate frames to a large awkward array and dump to
    a parquet file in the end
    """

    def __init__(self, context):
        """
        Setting up the module and defining inputs
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("muonitron_output_key", "xx", "Tracks")
        self.AddParameter("mcprimary_key", "xx", "MCPrimary")
        self.AddParameter("outfile", "xx", None)

    def Configure(self):
        """
        Getting the module parameters
        """
        self.muonitron_key = self.GetParameter("muonitron_output_key")
        self.mcprim_key = self.GetParameter("mcprimary_key")
        self.outfile = self.GetParameter("outfile")
        # builder = ak.ArrayBuilder()
        self.temp_list = []
        self.file_handle = open(self.outfile, "w")
        # self.temp_dict = defaultdict(list)

    def DAQ(self, frame):
        tracks = frame[self.muonitron_key]
        primary = frame[self.mcprim_key]
        data = { "primary_energy_zenith": [primary.energy, primary.dir.zenith, self.convert_type(primary)] }
        phis = [ random.uniform(0, 2 * math.pi) for _ in range(10000) ]
        depth_tracks_list = []
        for d, ts in tracks.items():
            if len(ts) > 0 and len(ts) <= 10000:
                # [depth, { f"track_{j}": [t.energy, t.radius, t.radius*math.sin(phis[j]), t.radius*math.cos(phis[j])] for j, t in enumerate(ts) }
                if len(ts) > 1000: print(len(ts))
                depth_tracks_list.append( [d, { f"track_{j}": [t.energy, t.radius, t.radius*math.sin(phis[j]), t.radius*math.cos(phis[j])] for j, t in enumerate(ts) } ])
        data.update({"data": depth_tracks_list})
        self.file_handle.write(json.dumps(data) + '\n')


    def Finish(self):
        # print(self.temp_list[0])
        # np.array(self.temp_list)
        self.file_handle.close()

    def convert_type(self, primary):
        if primary.type == dataclasses.I3Particle.PPlus:
            return [1, 0, 0, 0, 0]
        elif primary.type == dataclasses.I3Particle.He4Nucleus:
            return [0, 1, 0, 0, 0]
        elif primary.type == dataclasses.I3Particle.N14Nucleus:
            return [0, 0, 1, 0, 0]
        elif primary.type == dataclasses.I3Particle.Al27Nucleus:
            return [0, 0, 0, 1, 0]
        elif primary.type == dataclasses.I3Particle.Fe56Nucleus:
            return [0, 0, 0, 0, 1]
        else:
            print(primary.type)
            raise RuntimeError()

def icetray_script(argsparse):

   tray = I3Tray()

   MuonPropagator.set_seed(12345)
   crust = Crust(MuonPropagator("air", ecut=-1, vcut=5e-2, rho=0.673))
   crust.add_layer(Sphere(1948, 6374134),
                   MuonPropagator("ice", ecut=-1,
                                  vcut=5e-2, rho=0.832))
   crust.add_layer(Sphere(1748, 6373934),
                   MuonPropagator("ice", ecut=-1,
                                  vcut=5e-2, rho=1.005))


   tray.AddModule("I3Reader", "reader",
                  Filename=argsparse.infile)


   tray.AddModule('Muonitron', 'propagator',
                  Depths=list(np.linspace(args.mindepth, args.maxdepth, args.depthsteps)*I3Units.km),
                  Propagator=MuonPropagator("ice", ecut=-1, vcut=5e-2, rho=1.005),
                  Crust=crust,
                  MCTreeName="I3MCTree_preMuonProp")

   tray.AddModule(convert_muonitron_jsonl,
                  outfile=argsparse.outfile)

   # tray.Add("I3Writer","outwriter",
   #          filename=args.outfile)

   tray.Execute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='dump_muonitron_output')
    parser.add_argument("-i", "--infile", dest="infile", type=str, required=True)
    parser.add_argument("-o", "--outfile", dest="outfile", type=str)
    parser.add_argument("--mindepth", dest="mindepth", type=float, default=1.)
    parser.add_argument("--maxdepth", dest="maxdepth", type=float, default=2.8)
    parser.add_argument("--depthsteps", dest="depthsteps", type=int, default=100)
    args = parser.parse_args()

    icetray_script(args)
