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

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None

class convert_muonitron_hdf5(icetray.I3ConditionalModule):
    """
    Class to aggregate frames to an array and dump to
    a hdf5 file incrementally
    """
    def __init__(self, context):
        """
        Setting up the module and defining inputs
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("muonitron_output_key", "xx", "Tracks")
        self.AddParameter("mcprimary_key", "xx", "MCPrimary")
        self.AddParameter("outfile", "xx", None)
        self.AddParameter("buffer_size", "xx", 1000000)
    
    def Configure(self):
        """
        Getting the module parameters
        """
        self.muonitron_key = self.GetParameter("muonitron_output_key")
        self.mcprim_key = self.GetParameter("mcprimary_key")
        self.outfile = self.GetParameter("outfile")
        self.buffer_size = self.GetParameter("buffer_size")
        # builder = ak.ArrayBuilder()
        self.temp_list = []
        self.file_handle = h5py.File(self.outfile, "w")

        # Create Resizable Datasets
        # We don't know total size yet, so we make them resizable (maxshape=None)
        self.dset_prims = self.file_handle.create_dataset("primaries", 
                                                    shape=(0, 6), 
                                                    maxshape=(None, 7), dtype='f4', 
                                                    chunks=(10000, 7))
        self.dset_muons = self.file_handle.create_dataset("muons", 
                                                     shape=(0, 5), 
                                                     maxshape=(None, 6),
                                                     dtype='f4',
                                                     chunks=(100000, 6))
        self.dset_counts = self.file_handle.create_dataset("counts", 
                                                      shape=(0,), 
                                                      maxshape=(None,), 
                                                      dtype='i4', 
                                                      chunks=(10000,))
        self.buf_prim = []
        self.buf_muons = []
        self.buf_counts = []

    def DAQ(self, frame):
        tracks = frame[self.muonitron_key]
        primary = frame[self.mcprim_key]
        phis = [ random.uniform(0, 2 * math.pi) for _ in range(50000) ]
        for d, ts in tracks.items():
            self.buf_prim.append( [
                    primary.major_id, 
                    primary.minor_id,
                    primary.energy, 
                    primary.dir.zenith, 
                    primary.mass, 
                    primary.time,  
                    d
            ] )
            if len(ts) > 0:
                self.buf_muons.extend( [
                    [ primary.major_id, 
                     primary. minor_id, 
                     t.energy, 
                     t.radius*math.sin(phis[j]), 
                     t.radius*math.cos(phis[j]), 
                     t.time] for j, t in enumerate(ts)
                ] )
                self.buf_counts.append(len(ts))
            else:
                self.buf_muons.extend( [ [primary.major_id, 
                                          primary.minor_id, 
                                          0, 0, 0, 0] ] )
                self.buf_counts.append(1)
        if len(self.buf_prim) >= self.buffer_size:
            self._append_datasets()
            self.buf_prim = []
            self.buf_muons = []
            self.buf_counts = []

        
    def Finish(self):
        self._append_datasets()
        self.file_handle.close()
        with h5py.File(self.outfile, "r") as f:
            print(f.keys())
            print(f['primaries'].shape)
            print(f['primaries'][0:100])
            print(f["muons"][0:100])
    
    def _append_datasets(self):
        # Current size
        curr_p = self.dset_prims.shape[0]
        curr_m = self.dset_muons.shape[0]
        curr_n = self.dset_counts.shape[0]
        
        # New size
        new_p = curr_p + len(self.buf_prim)
        new_m = curr_m + len(self.buf_muons)
        new_n = curr_n + len(self.buf_counts)

        # Resize
        self.dset_prims.resize((new_p, 7))
        self.dset_muons.resize((new_m, 6))
        self.dset_counts.resize((new_n,))

        self.dset_prims[curr_p:] = self.buf_prim
        self.dset_muons[curr_m:] = self.buf_muons
        self.dset_counts[curr_n:] = self.buf_counts


class convert_muonitron_parquet(icetray.I3ConditionalModule):
    """Dump Muonitron output to Parquet incrementally using PyArrow.

    Output schema is compatible with `training/hf_dataloader.py`.

    IDs are stored exactly as int64 columns:
    - primary_major_id: int64
    - primary_minor_id: int64

    Physics features stay in the nested float lists expected by the HF loader:
    - primary: list[float32] (len=4) -> [E_GeV, zenith_rad, mass_A, depth]
    - muons: list[list[float32]] -> each muon row is [E_GeV, x_m, y_m]

    The loader can optionally re-attach IDs to the tensors for training-time
    sanity checks.
    """

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("muonitron_output_key", "xx", "Tracks")
        self.AddParameter("mcprimary_key", "xx", "MCPrimary")
        self.AddParameter("outfile", "xx", None)
        self.AddParameter("buffer_size", "xx", 10000)

    def Configure(self):
        if pa is None or pq is None:
            raise ImportError("pyarrow is required for Parquet output. Install with: pip install pyarrow")

        self.muonitron_key = self.GetParameter("muonitron_output_key")
        self.mcprim_key = self.GetParameter("mcprimary_key")
        self.outfile = self.GetParameter("outfile")
        self.buffer_size = int(self.GetParameter("buffer_size"))

        if not self.outfile:
            raise ValueError("outfile must be provided")

        # PyArrow can write to local paths (and some supported filesystem URIs).
        # If a custom URI scheme is used (e.g. pelican://), ParquetWriter will fail.
        if "://" in self.outfile and not self.outfile.startswith("file://"):
            raise ValueError(
                f"Unsupported outfile URI for pyarrow ParquetWriter: {self.outfile}. "
                "Use a local path, or write via a custom filesystem handle (e.g. pelicanfs)."
            )

        if os.path.isdir(self.outfile):
            raise ValueError(f"outfile must be a file path, got directory: {self.outfile}")

        outdir = os.path.dirname(self.outfile)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        self.schema = pa.schema([
            ("primary_major_id", pa.uint64()),
            ("primary_minor_id", pa.uint64()),
            ("primary", pa.list_(pa.float32())),
            ("muons", pa.list_(pa.list_(pa.float32())))
        ])
        try:
            self.writer = pq.ParquetWriter(self.outfile, self.schema)
        except Exception as e:
            raise RuntimeError(f"Failed to open Parquet outfile for writing: {self.outfile}: {e}") from e

        self.buf_primary_major_id = []
        self.buf_primary_minor_id = []
        self.buf_primary = []
        self.buf_muons = []

    def DAQ(self, frame):
        tracks = frame[self.muonitron_key]
        primary = frame[self.mcprim_key]

        # These IDs can exceed signed int64 / platform C long.
        # Store as uint64 and buffer as numpy scalars to avoid PyArrow overflow.
        major_id = np.uint64(primary.major_id)
        minor_id = np.uint64(primary.minor_id)

        for depth, ts in tracks.items():
            if len(ts) == 0:
                continue

            self.buf_primary_major_id.append(major_id)
            self.buf_primary_minor_id.append(minor_id)
            self.buf_primary.append([
                float(primary.energy),
                float(primary.dir.zenith),
                float(primary.mass),
                float(depth),
            ])
            self.buf_muons.append([])

            mu_rows = []
            for t in ts:
                phi = random.uniform(0, 2 * math.pi)
                mu_rows.append([
                    float(t.energy),
                    float(t.radius * math.sin(phi)),
                    float(t.radius * math.cos(phi)),
                ])
            self.buf_muons.append(mu_rows)

        if len(self.buf_primary) >= self.buffer_size:
            self._flush()

    def Finish(self):
        self._flush()
        if getattr(self, "writer", None) is not None:
            self.writer.close()

    def _flush(self):
        if not self.buf_primary:
            return

        table = pa.Table.from_pydict(
            {
                "primary_major_id": self.buf_primary_major_id,
                "primary_minor_id": self.buf_primary_minor_id,
                "primary": self.buf_primary,
                "muons": self.buf_muons,
            },
            schema=self.schema,
        )
        self.writer.write_table(table)
        self.buf_primary_major_id = []
        self.buf_primary_minor_id = []
        self.buf_primary = []
        self.buf_muons = []

class convert_muonitron_jsonl(icetray.I3ConditionalModule):
    """
    Class to dump per frame information into per line of a file. Frame information is summarized in a json object
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
        self.hdf5_file_handle = h5py.FILE(self.outfile, "w")
        # self.temp_dict = defaultdict(list)

    def DAQ(self, frame):
        tracks = frame[self.muonitron_key]
        primary = frame[self.mcprim_key]
        data = { "primary_energy_zenith": [primary.energy, 
                                           primary.dir.zenith, 
                                           primary.mass] }
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
                  Depths=list(np.linspace(argsparse.mindepth, argsparse.maxdepth, argsparse.depthsteps)*I3Units.km),
                  Propagator=MuonPropagator("ice", ecut=-1, vcut=5e-2, rho=1.005),
                  Crust=crust,
                  MCTreeName="I3MCTree_preMuonProp")

#    tray.AddModule(convert_muonitron_jsonl,
#                   outfile=argsparse.outfile)
   
   if argsparse.format == "parquet":
       tray.AddModule(convert_muonitron_parquet,
                      outfile=argsparse.outfile,
                      buffer_size=argsparse.buffer_size)
   else:
       tray.AddModule(convert_muonitron_hdf5,
                      outfile=argsparse.outfile,
                      buffer_size=argsparse.buffer_size)

   tray.Execute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='dump_muonitron_output')
    parser.add_argument("-i", "--infile", dest="infile", type=str, required=True)
    parser.add_argument("-o", "--outfile", dest="outfile", type=str)
    parser.add_argument("--format", dest="format", choices=["hdf5", "parquet"], default="hdf5")
    parser.add_argument("--buffer-size", dest="buffer_size", type=int, default=10000)
    parser.add_argument("--mindepth", dest="mindepth", type=float, default=1.)
    parser.add_argument("--maxdepth", dest="maxdepth", type=float, default=2.8)
    parser.add_argument("--depthsteps", dest="depthsteps", type=int, default=100)
    args = parser.parse_args()

    icetray_script(args)
