# cr_muons

First attempt will be a regressor to get the number of muons in a bundle given a primary (E_primary, Mass_primary, zenith_primary, and slant depth) then a conditional wasserstein gan will take over to learn the kinematics from primary (E_primary, Mass_primary, zenith_primary, and slant depth) and output muons

## Normalization

* We want to be as close to the range of [-1, 1] for most of the values to make the NN happy. 
* Normalization for Primary
** Energy - log10(E_primary_PeV) - We picked PeV because we range from 10^0 to 10^9 GeV
** Zenith - cos(zenith_rads)
** Mass - log10(Mass_primary_GeV)
** Slant Depth - Slant depth in km
* Normalization for Muon
** Same as primary - log10(Muon_PeV)
** X, Y - Coordinates around the shower axis divided by 500 m
