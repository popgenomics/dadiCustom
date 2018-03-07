#! ~/py33/bin
# -*- coding: utf-8 -*-

# CRoux version
# Heliconius

"""module modeledemo contenant les différents modèles démographiques de divergence"""

import numpy

import site
#site.addsitedir('~/Dropbox/Dadi_tuto/SOURCE/dadi-1.6.3_modif')
import dadi

def PAN(params, (n1,n2), pts):
    nu1, Ts = params
    """
    Panmixia

    nu1: Size of all populations (ancestral + current ones)
    Ts: here to avoid bug (needs more than one parameter). Estimated in [0, 0.00001]
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx, nu=nu1)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu1, m12=20000, m21=20000)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs


def PANex(params, (n1,n2), pts):
    nu1a, nu1, Te = params
    """
    Model PAN with population expansion.
    nu1a: Size of population 1 after split.
    nu1: Size of population 1 after split.
    Te: Time of the exponential growth in strict isolation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx, nu=nu1a)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    phi = dadi.Integration.one_pop(phi, xx, Te, nu1_func)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2
    phi = dadi.Integration.two_pops(phi, xx, 0, nu1a, nu1a, m12=20000, m21=20000)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs

def PAN2Nex(params, (n1,n2), pts):
    nu1a, nu1, nr, bf, Te = params
    """
    Panmixia with hetero_N and expansion
    
    nu1: Size of population 1 after split.
    n1,n2: Size of fs to generate.
    nr: Proportion of non-recombining regions
    bf : Background factor (to which extent the effective population size is reduced in the non-recombining regions)
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # Spectrum of non-recombining regions
    # phi for the equilibrium ancestral population
    phinr = dadi.PhiManip.phi_1D(xx, nu=nu1a*bf)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1*bf) * t/Te)
    phinr = dadi.Integration.one_pop(phinr, xx, Te, nu1_func)
    # Now do the divergence event
    phinr = dadi.PhiManip.phi_1D_to_2D(xx, phinr)
    # We set the population sizes after the split to nu1 and nu2	
    phinr = dadi.Integration.two_pops(phinr, xx, 0, nu1*bf, nu1*bf, m12=20000, m21=20000)
    # Finally, calculate the spectrum.
    fsnr = dadi.Spectrum.from_phi(phinr, (n1,n2), (xx,xx))
    
    # Spectrum of recombining regions
    # phi for the equilibrium ancestral population
    phir = dadi.PhiManip.phi_1D(xx, nu=nu1a)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    phir = dadi.Integration.one_pop(phir, xx, Te, nu1_func)
    # Now do the divergence event
    phir = dadi.PhiManip.phi_1D_to_2D(xx, phir)
    # We set the population sizes after the split to nu1 and nu2
    phir = dadi.Integration.two_pops(phir, xx, 0, nu1, nu1, m12=20000, m21=20000)
    # Finally, calculate the spectrum.
    fsr = dadi.Spectrum.from_phi(phir, (n1,n2), (xx,xx))
    ### Sum the two spectra
    fs= nr*fsnr + (1-nr)*fsr
    return fs


def PAN2N(params, (n1,n2), pts):
    nu1, nr, bf = params
    """
    Panmixia with hetero_N
    
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    Ts: The scaled time of the split
    n1,n2: Size of fs to generate.
    nr: Proportion of non-recombining regions
    bf : Background factor (to which extent the effective population size is reduced in the non-recombining regions)
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # Spectrum of non-recombining regions
    # phi for the equilibrium ancestral population
    phinr = dadi.PhiManip.phi_1D(xx, nu=nu1*bf)
    # Now do the divergence event
    phinr = dadi.PhiManip.phi_1D_to_2D(xx, phinr)
    # We set the population sizes after the split to nu1 and nu2	
    phinr = dadi.Integration.two_pops(phinr, xx, 0, nu1*bf, nu1*bf, m12=20000, m21=20000)
    # Finally, calculate the spectrum.
    fsnr = dadi.Spectrum.from_phi(phinr, (n1,n2), (xx,xx))
    
    # Spectrum of recombining regions
    # phi for the equilibrium ancestral population
    phir = dadi.PhiManip.phi_1D(xx, nu=nu1)
    # Now do the divergence event
    phir = dadi.PhiManip.phi_1D_to_2D(xx, phir)
    # We set the population sizes after the split to nu1 and nu2
    phir = dadi.Integration.two_pops(phir, xx, 0, nu1, nu1, m12=20000, m21=20000)
    # Finally, calculate the spectrum.
    fsr = dadi.Spectrum.from_phi(phir, (n1,n2), (xx,xx))
    ### Sum the two spectra
    fs= nr*fsnr + (1-nr)*fsr
    return fs


def SI(params, (n1,n2), pts):
    nu1, nu2, Ts = params
    """
    Model with split and complete isolation.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs


def SIex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, Ts, Te = params
    """
    Model with split and complete isolation.
    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Te: Time of the exponential growth in strict isolation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # Exponential growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs
 
def SI2N(params, (n1,n2), pts):
    nu1, nu2, Ts, nr, bf = params
    """
    Model with split and complete isolation, heterogenous effective population size (2 classes, shared by the two populations = background selection)

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    Ts: The scaled time of the split
    n1,n2: Size of fs to generate.
    nr: Proportion of non-recombining regions
    bf : Background factor (to which extent the effective population size is reduced in the non-recombining regions)
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # Spectrum of non-recombining regions
    # phi for the equilibrium ancestral population
    phinr = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phinr = dadi.PhiManip.phi_1D_to_2D(xx, phinr)
    # We set the population sizes after the split to nu1 and nu2	
    phinr = dadi.Integration.two_pops(phinr, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    # Finally, calculate the spectrum.
    fsnr = dadi.Spectrum.from_phi(phinr, (n1,n2), (xx,xx))
    # Spectrum of recombining regions
    # phi for the equilibrium ancestral population
    phir = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phir = dadi.PhiManip.phi_1D_to_2D(xx, phir)
    # We set the population sizes after the split to nu1 and nu2
    phir = dadi.Integration.two_pops(phir, xx, Ts, nu1, nu2, m12=0, m21=0)
    # Finally, calculate the spectrum.
    fsr = dadi.Spectrum.from_phi(phir, (n1,n2), (xx,xx))
    ### Sum the two spectra
    fs= nr*fsnr + (1-nr)*fsr
    return fs

def SI2Nex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, Ts, Te, nr, bf = params
    """
    Model with split and complete isolation, heterogenous effective population size (2 classes, shared by the two populations = background selection)
    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    Ts: The scaled time of the split
    Te: Time of the exponential growth in strict isolation.
    n1,n2: Size of fs to generate.
    nr: Proportion of non-recombining regions
    bf : Background factor (to which extent the effective population size is reduced in the non-recombining regions)
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # Spectrum of non-recombining regions
    # phi for the equilibrium ancestral population
    phinr = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phinr = dadi.PhiManip.phi_1D_to_2D(xx, phinr)
    # We set the population sizes after the split to nu1 and nu2	
    phinr = dadi.Integration.two_pops(phinr, xx, Ts, nu1a*bf, nu2a*bf, m12=0, m21=0)
    # Exponential growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phinr = dadi.Integration.two_pops(phinr, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # Finally, calculate the spectrum.
    fsnr = dadi.Spectrum.from_phi(phinr, (n1,n2), (xx,xx))
   
    # Spectrum of recombining regions
    # phi for the equilibrium ancestral population
    phir = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phir = dadi.PhiManip.phi_1D_to_2D(xx, phir)
    # We set the population sizes after the split to nu1 and nu2
    phir = dadi.Integration.two_pops(phir, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # Exponential growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phir = dadi.Integration.two_pops(phir, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # Finally, calculate the spectrum.
    fsr = dadi.Spectrum.from_phi(phir, (n1,n2), (xx,xx))
    ### Sum the two spectra
    fs= nr*fsnr + (1-nr)*fsr
    return fs


def IM(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts = params
    """
    Model with migration during the divergence.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs


def IMex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Te = params
    """
    Model with migration during the divergence.
    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Te: Time of the exponential growth in strict isolation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # Exponential growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs
	
def AM(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tam = params
    """
    Model with split, ancient migration

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21 
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1, nu2, m12=0, m21=0)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))

    return fs

def AMex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tam, Te = params
    """
    Model with split, ancient migration
    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    Te: Time of the exponential growth in strict isolation.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21 
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1, nu2, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs

def PAM(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tam = params
    """
    Model with split, followed by two periods of ancient migration

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21 
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1, nu2, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21 
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1, nu2, m12=0, m21=0)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs
   
def PAMex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tam, Te = params
    """
    Model with split, followed by two periods of ancient migration

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21 
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21 
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phi = dadi.Integration.two_pops(phi, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs

def SC(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tsc = params
    """
    Model with split, complete isolation, followed by secondary contact

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1, nu2, m12=m12, m21=m21)

    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs

def SCex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te = params
    """
    Model with split, complete isolation, followed by secondary contact
    nu1a: Size of population 1 after split.
    nu2a: Size of population 2 after split.
    nu1: Size of population 1.
    nu2: Size of population 2.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    Te: Time of the exponential growth in continuous migration.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs

def PSC(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tsc = params
    """
    Model with split, complete isolation, followed by two periods of secondary contact

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1, nu2, m12=m12, m21=m21)

    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs


def PSCex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te = params
    """
    Model with split, complete isolation, followed by two periods of secondary contact

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)
    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phi = dadi.Integration.two_pops(phi, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phi = dadi.Integration.two_pops(phi, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phi = dadi.Integration.two_pops(phi, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21)
    # Finally, calculate the spectrum.
    fs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return fs

def IM2M(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, P = params
    """
    Model with migration during the divergence with two type of migration

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    P: The proportion of the genome evolving neutrally
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum
    # phi for the equilibrium ancestral population
    phiN = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN = dadi.PhiManip.phi_1D_to_2D(xx, phiN)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN = dadi.Spectrum.from_phi(phiN, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum
    # phi for the equilibrium ancestral population
    phiI = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI = dadi.PhiManip.phi_1D_to_2D(xx, phiI)
    # We set the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI = dadi.Spectrum.from_phi(phiI, (n1,n2), (xx,xx))

    ### Sum the two spectra
    fs = P*fsN + (1-P)*fsI
    return fs

def IM2Mex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Te, P = params
    """
    Model with migration during the divergence with two type of migration

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    P: The proportion of the genome evolving neutrally
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum
    # phi for the equilibrium ancestral population
    phiN = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN = dadi.PhiManip.phi_1D_to_2D(xx, phiN)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN = dadi.Integration.two_pops(phiN, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN = dadi.Spectrum.from_phi(phiN, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum
    # phi for the equilibrium ancestral population
    phiI = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI = dadi.PhiManip.phi_1D_to_2D(xx, phiI)
    # We set the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI = dadi.Integration.two_pops(phiI, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21)
    # calculate the spectrum.
    fsI = dadi.Spectrum.from_phi(phiI, (n1,n2), (xx,xx))

    ### Sum the two spectra
    fs = P*fsN + (1-P)*fsI
    return fs

def AM2M(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tam, P = params
    """
    Model of semi permeability with split, complete isolation, followed by ancient migration with 2 migration rates

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    P: The proportion of the genome evolving neutrally
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum
    # phi for the equilibrium ancestral population
    phiN = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN = dadi.PhiManip.phi_1D_to_2D(xx, phiN)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsN = dadi.Spectrum.from_phi(phiN, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum
    # phi for the equilibrium ancestral population
    phiI = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI = dadi.PhiManip.phi_1D_to_2D(xx, phiI)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI = dadi.Spectrum.from_phi(phiI, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P*fsN + (1-P)*fsI
    return fs


def AM2Mex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tam, Te, P = params
    """
    Model of semi permeability with split, complete isolation, followed by ancient migration with 2 migration rates

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    P: The proportion of the genome evolving neutrally
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum
    # phi for the equilibrium ancestral population
    phiN = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN = dadi.PhiManip.phi_1D_to_2D(xx, phiN)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN = dadi.Integration.two_pops(phiN, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsN = dadi.Spectrum.from_phi(phiN, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum
    # phi for the equilibrium ancestral population
    phiI = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI = dadi.PhiManip.phi_1D_to_2D(xx, phiI)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI = dadi.Integration.two_pops(phiI, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI = dadi.Spectrum.from_phi(phiI, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P*fsN + (1-P)*fsI
    return fs


def PAM2M(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tam, P = params
    """
    Model of semi permeability with split, complete isolation, followed by two periods of ancient migration with 2 migration rates

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    P: The proportion of the genome evolving neutrally
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum
    # phi for the equilibrium ancestral population
    phiN = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN = dadi.PhiManip.phi_1D_to_2D(xx, phiN)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Tam, nu1, nu2, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsN = dadi.Spectrum.from_phi(phiN, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum
    # phi for the equilibrium ancestral population
    phiI = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI = dadi.PhiManip.phi_1D_to_2D(xx, phiI)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Tam, nu1, nu2, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI = dadi.Spectrum.from_phi(phiI, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P*fsN + (1-P)*fsI
    return fs


def PAM2Mex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tam, Te, P = params
    """
    Model of semi permeability with split, complete isolation, followed by two periods of ancient migration with 2 migration rates

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    P: The proportion of the genome evolving neutrally
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll usr
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum
    # phi for the equilibrium ancestral population
    phiN = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN = dadi.PhiManip.phi_1D_to_2D(xx, phiN)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN = dadi.Integration.two_pops(phiN, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsN = dadi.Spectrum.from_phi(phiN, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum
    # phi for the equilibrium ancestral population
    phiI = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI = dadi.PhiManip.phi_1D_to_2D(xx, phiI)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI = dadi.Integration.two_pops(phiI, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI = dadi.Spectrum.from_phi(phiI, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P*fsN + (1-P)*fsI
    return fs


def SC2M(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tsc, P = params
    """
    Model of semi permeability with split, complete isolation, followed by secondary contact with 2 migration rates

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    P: The proportion of the genome evolving neutrally
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum
    # phi for the equilibrium ancestral population
    phiN = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN = dadi.PhiManip.phi_1D_to_2D(xx, phiN)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN = dadi.Spectrum.from_phi(phiN, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum
    # phi for the equilibrium ancestral population
    phiI = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI = dadi.PhiManip.phi_1D_to_2D(xx, phiI)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Tsc, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI = dadi.Spectrum.from_phi(phiI, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P*fsN + (1-P)*fsI
    return fs


def SC2Mex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te, P = params
    """
    Model of semi permeability with split, complete isolation, followed by secondary contact with 2 migration rates

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    P: The proportion of the genome evolving neutrally
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum
    # phi for the equilibrium ancestral population
    phiN = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN = dadi.PhiManip.phi_1D_to_2D(xx, phiN)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN = dadi.Integration.two_pops(phiN, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN = dadi.Spectrum.from_phi(phiN, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum
    # phi for the equilibrium ancestral population
    phiI = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI = dadi.PhiManip.phi_1D_to_2D(xx, phiI)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Tsc, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI = dadi.Integration.two_pops(phiI, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI = dadi.Spectrum.from_phi(phiI, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P*fsN + (1-P)*fsI
    return fs


def PSC2M(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tsc, P = params
    """
    Model of semi permeability with split, complete isolation, followed by two periods of secondary contact with 2 migration rates and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    P: The proportion of the genome evolving neutrally
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum
    # phi for the equilibrium ancestral population
    phiN = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN = dadi.PhiManip.phi_1D_to_2D(xx, phiN)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN = dadi.Spectrum.from_phi(phiN, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum
    # phi for the equilibrium ancestral population
    phiI = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI = dadi.PhiManip.phi_1D_to_2D(xx, phiI)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Tsc, nu1, nu2, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Tsc, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI = dadi.Spectrum.from_phi(phiI, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P*fsN + (1-P)*fsI
    return fs


def PSC2Mex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te, P = params
    """
    Model of semi permeability with split, complete isolation, followed by two periods of secondary contact with 2 migration rates and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    P: The proportion of the genome evolving neutrally
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum
    # phi for the equilibrium ancestral population
    phiN = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN = dadi.PhiManip.phi_1D_to_2D(xx, phiN)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN = dadi.Integration.two_pops(phiN, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN = dadi.Integration.two_pops(phiN, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN = dadi.Integration.two_pops(phiN, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN = dadi.Spectrum.from_phi(phiN, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum
    # phi for the equilibrium ancestral population
    phiI = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI = dadi.PhiManip.phi_1D_to_2D(xx, phiI)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Tsc, nu1a, nu2a, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI = dadi.Integration.two_pops(phiI, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI = dadi.Integration.two_pops(phiI, xx, Tsc, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI = dadi.Integration.two_pops(phiI, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI = dadi.Spectrum.from_phi(phiI, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P*fsN + (1-P)*fsI
    return fs

 
def IM2M2P(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, P1, P2 = params
    """
    Model with migration during the divergence with two type of migration and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    P1: The proportion of the genome evolving neutrally in population 1
    P2: The proportion of the genome evolving neutrally in population 2
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiN1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1N2)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN1N2 = dadi.Spectrum.from_phi(phiN1N2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiI1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1I2)
    # We set the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI1I2 = dadi.Spectrum.from_phi(phiI1I2, (n1,n2), (xx,xx))

    ### Calculate the neutral spectrum in population 1 and the genomic island spectrum in population 2
    # phi for the equilibrium ancestral population
    phiN1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1I2)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1, nu2, m12=m12, m21=0)
    # calculate the spectrum.
    fsN1I2 = dadi.Spectrum.from_phi(phiN1I2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and the neutral spectrum in population 2
    # phi for the equilibrium ancestral population
    phiI1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1N2)
    # We set the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1, nu2, m12=0, m21=m21)
    # calculate the spectrum.
    fsI1N2 = dadi.Spectrum.from_phi(phiI1N2, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P1*P2*fsN1N2 + (1-P1)*(1-P2)*fsI1I2 + P1*(1-P2)*fsN1I2 + (1-P1)*P2*fsI1N2
    return fs


def IM2M2Pex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Te, P1, P2 = params
    """
    Model with migration during the divergence with two type of migration and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    P1: The proportion of the genome evolving neutrally in population 1
    P2: The proportion of the genome evolving neutrally in population 2
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiN1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1N2)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN1N2 = dadi.Spectrum.from_phi(phiN1N2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiI1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1I2)
    # We set the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI1I2 = dadi.Spectrum.from_phi(phiI1I2, (n1,n2), (xx,xx))

    ### Calculate the neutral spectrum in population 1 and the genomic island spectrum in population 2
    # phi for the equilibrium ancestral population
    phiN1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1I2)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1a, nu2a, m12=m12, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Te, nu1_func, nu2_func, m12=m12, m21=0)
    # calculate the spectrum.
    fsN1I2 = dadi.Spectrum.from_phi(phiN1I2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and the neutral spectrum in population 2
    # phi for the equilibrium ancestral population
    phiI1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1N2)
    # We set the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1a, nu2a, m12=0, m21=m21)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Te, nu1_func, nu2_func, m12=0, m21=m21)
    # calculate the spectrum.
    fsI1N2 = dadi.Spectrum.from_phi(phiI1N2, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P1*P2*fsN1N2 + (1-P1)*(1-P2)*fsI1I2 + P1*(1-P2)*fsN1I2 + (1-P1)*P2*fsI1N2
    return fs


def AM2M2P(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tam, P1, P2 = params
    """
    Model of semi permeability with split, complete isolation, followed by ancient migration with 2 migration rates and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    P1: The proportion of the genome evolving neutrally in population 1
    P2: The proportion of the genome evolving neutrally in population 2
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiN1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsN1N2 = dadi.Spectrum.from_phi(phiN1N2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiI1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI1I2 = dadi.Spectrum.from_phi(phiI1I2, (n1,n2), (xx,xx))

    ### Calculate the neutral spectrum in population 1 and the genomic island spectrum in population 2
    # phi for the equilibrium ancestral population
    phiN1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1, nu2, m12=m12, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsN1I2 = dadi.Spectrum.from_phi(phiN1I2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and the neutral spectrum in population 2
    # phi for the equilibrium ancestral population
    phiI1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1, nu2, m12=0, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI1N2 = dadi.Spectrum.from_phi(phiI1N2, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P1*P2*fsN1N2 + (1-P1)*(1-P2)*fsI1I2 + P1*(1-P2)*fsN1I2 + (1-P1)*P2*fsI1N2
    return fs

def AM2M2Pex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tam, Te, P1, P2 = params
    """
    Model of semi permeability with split, complete isolation, followed by ancient migration with 2 migration rates and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    P1: The proportion of the genome evolving neutrally in population 1
    P2: The proportion of the genome evolving neutrally in population 2
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiN1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsN1N2 = dadi.Spectrum.from_phi(phiN1N2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiI1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI1I2 = dadi.Spectrum.from_phi(phiI1I2, (n1,n2), (xx,xx))

    ### Calculate the neutral spectrum in population 1 and the genomic island spectrum in population 2
    # phi for the equilibrium ancestral population
    phiN1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1a, nu2a, m12=m12, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsN1I2 = dadi.Spectrum.from_phi(phiN1I2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and the neutral spectrum in population 2
    # phi for the equilibrium ancestral population
    phiI1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1a, nu2a, m12=0, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI1N2 = dadi.Spectrum.from_phi(phiI1N2, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P1*P2*fsN1N2 + (1-P1)*(1-P2)*fsI1I2 + P1*(1-P2)*fsN1I2 + (1-P1)*P2*fsI1N2
    return fs

def PAM2M2P(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tam, P1, P2 = params
    """
    Model of semi permeability with split, complete isolation, followed by two periods of ancient migration with 2 migration rates and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    P1: The proportion of the genome evolving neutrally in population 1
    P2: The proportion of the genome evolving neutrally in population 2
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiN1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsN1N2 = dadi.Spectrum.from_phi(phiN1N2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiI1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI1I2 = dadi.Spectrum.from_phi(phiI1I2, (n1,n2), (xx,xx))

    ### Calculate the neutral spectrum in population 1 and the genomic island spectrum in population 2
    # phi for the equilibrium ancestral population
    phiN1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1, nu2, m12=m12, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1, nu2, m12=m12, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsN1I2 = dadi.Spectrum.from_phi(phiN1I2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and the neutral spectrum in population 2
    # phi for the equilibrium ancestral population
    phiI1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1, nu2, m12=0, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1, nu2, m12=0, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tam, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI1N2 = dadi.Spectrum.from_phi(phiI1N2, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P1*P2*fsN1N2 + (1-P1)*(1-P2)*fsI1I2 + P1*(1-P2)*fsN1I2 + (1-P1)*P2*fsI1N2
    return fs

def PAM2M2Pex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tam, Te, P1, P2 = params
    """
    Model of semi permeability with split, complete isolation, followed by two periods of ancient migration with 2 migration rates and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the ancient migration (in units of 2*Na generations).
    Tam: The scale time between the ancient migration and present.
    P1: The proportion of the genome evolving neutrally in population 1
    P2: The proportion of the genome evolving neutrally in population 2
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiN1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1a, nu2a, m12=m12, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsN1N2 = dadi.Spectrum.from_phi(phiN1N2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiI1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI1I2 = dadi.Spectrum.from_phi(phiI1I2, (n1,n2), (xx,xx))

    ### Calculate the neutral spectrum in population 1 and the genomic island spectrum in population 2
    # phi for the equilibrium ancestral population
    phiN1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1a, nu2a, m12=m12, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1a, nu2a, m12=m12, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsN1I2 = dadi.Spectrum.from_phi(phiN1I2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and the neutral spectrum in population 2
    # phi for the equilibrium ancestral population
    phiI1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1a, nu2a, m12=0, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1a, nu2a, m12=0, m21=m21)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tam, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI1N2 = dadi.Spectrum.from_phi(phiI1N2, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P1*P2*fsN1N2 + (1-P1)*(1-P2)*fsI1I2 + P1*(1-P2)*fsN1I2 + (1-P1)*P2*fsI1N2
    return fs

def SC2M2P(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tsc, P1, P2 = params
    """
    Model of semi permeability with split, complete isolation, followed by secondary contact with 2 migration rates and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    P1: The proportion of the genome evolving neutrally in population 1
    P2: The proportion of the genome evolving neutrally in population 2
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiN1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN1N2 = dadi.Spectrum.from_phi(phiN1N2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiI1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tsc, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI1I2 = dadi.Spectrum.from_phi(phiI1I2, (n1,n2), (xx,xx))

    ### Calculate the neutral spectrum in population 1 and the genomic island spectrum in population 2
    # phi for the equilibrium ancestral population
    phiN1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tsc, nu1, nu2, m12=m12, m21=0)
    # calculate the spectrum.
    fsN1I2 = dadi.Spectrum.from_phi(phiN1I2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and the neutral spectrum in population 2
    # phi for the equilibrium ancestral population
    phiI1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tsc, nu1, nu2, m12=0, m21=m21)
    # calculate the spectrum.
    fsI1N2 = dadi.Spectrum.from_phi(phiI1N2, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P1*P2*fsN1N2 + (1-P1)*(1-P2)*fsI1I2 + P1*(1-P2)*fsN1I2 + (1-P1)*P2*fsI1N2
    return fs

def SC2M2Pex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te, P1, P2 = params
    """
    Model of semi permeability with split, complete isolation, followed by secondary contact with 2 migration rates and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    P1: The proportion of the genome evolving neutrally in population 1
    P2: The proportion of the genome evolving neutrally in population 2
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiN1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN1N2 = dadi.Spectrum.from_phi(phiN1N2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiI1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tsc, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI1I2 = dadi.Spectrum.from_phi(phiI1I2, (n1,n2), (xx,xx))

    ### Calculate the neutral spectrum in population 1 and the genomic island spectrum in population 2
    # phi for the equilibrium ancestral population
    phiN1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tsc, nu1a, nu2a, m12=m12, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Te, nu1_func, nu2_func, m12=m12, m21=0)
    # calculate the spectrum.
    fsN1I2 = dadi.Spectrum.from_phi(phiN1I2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and the neutral spectrum in population 2
    # phi for the equilibrium ancestral population
    phiI1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tsc, nu1a, nu2a, m12=0, m21=m21)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Te, nu1_func, nu2_func, m12=0, m21=m21)
    # calculate the spectrum.
    fsI1N2 = dadi.Spectrum.from_phi(phiI1N2, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P1*P2*fsN1N2 + (1-P1)*(1-P2)*fsI1I2 + P1*(1-P2)*fsN1I2 + (1-P1)*P2*fsI1N2
    return fs

def PSC2M2P(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tsc, P1, P2 = params
    """
    Model of semi permeability with split, complete isolation, followed by two periods of secondary contact with 2 migration rates and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    P1: The proportion of the genome evolving neutrally in population 1
    P2: The proportion of the genome evolving neutrally in population 2
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiN1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    # calculate the spectrum.
    fsN1N2 = dadi.Spectrum.from_phi(phiN1N2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiI1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tsc, nu1, nu2, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tsc, nu1, nu2, m12=0, m21=0)
    # calculate the spectrum.
    fsI1I2 = dadi.Spectrum.from_phi(phiI1I2, (n1,n2), (xx,xx))

    ### Calculate the neutral spectrum in population 1 and the genomic island spectrum in population 2
    # phi for the equilibrium ancestral population
    phiN1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tsc, nu1, nu2, m12=m12, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tsc, nu1, nu2, m12=m12, m21=0)
    # calculate the spectrum.
    fsN1I2 = dadi.Spectrum.from_phi(phiN1I2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and the neutral spectrum in population 2
    # phi for the equilibrium ancestral population
    phiI1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tsc, nu1, nu2, m12=0, m21=m21)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tsc, nu1, nu2, m12=0, m21=m21) 
    # calculate the spectrum.
    fsI1N2 = dadi.Spectrum.from_phi(phiI1N2, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P1*P2*fsN1N2 + (1-P1)*(1-P2)*fsI1I2 + P1*(1-P2)*fsN1I2 + (1-P1)*P2*fsI1N2
    return fs

def PSC2M2Pex(params, (n1,n2), pts):
    nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te, P1, P2 = params
    """
    Model of semi permeability with split, complete isolation, followed by two periods of secondary contact with 2 migration rates and two proportions

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from pop 2 to pop 1 (2*Na*m12).
    m21: Migration from pop 1 to pop 2.
    0: Effective migration from pop 2 to pop 1 in genomic islands.
    0: Effective migration from pop 1 to pop 2 in genomic islands.
    Ts: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    Tsc: The scale time between the secondary contact and present.
    P1: The proportion of the genome evolving neutrally in population 1
    P2: The proportion of the genome evolving neutrally in population 2
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    ### Calculate the neutral spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiN1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and m21
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Tsc, nu1a, nu2a, m12=m12, m21=m21)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN1N2 = dadi.Integration.two_pops(phiN1N2, xx, Te, nu1_func, nu2_func, m12=m12, m21=m21) # CRoux 10/07/2017
    # calculate the spectrum.
    fsN1N2 = dadi.Spectrum.from_phi(phiN1N2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and 2
    # phi for the equilibrium ancestral population
    phiI1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tsc, nu1a, nu2a, m12=0, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and 0
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Tsc, nu1a, nu2a, m12=0, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI1I2 = dadi.Integration.two_pops(phiI1I2, xx, Te, nu1_func, nu2_func, m12=0, m21=0)
    # calculate the spectrum.
    fsI1I2 = dadi.Spectrum.from_phi(phiI1I2, (n1,n2), (xx,xx))

    ### Calculate the neutral spectrum in population 1 and the genomic island spectrum in population 2
    # phi for the equilibrium ancestral population
    phiN1I2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phiN1I2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tsc, nu1a, nu2a, m12=m12, m21=0)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to m12 and 0
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Tsc, nu1a, nu2a, m12=m12, m21=0)
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiN1I2 = dadi.Integration.two_pops(phiN1I2, xx, Te, nu1_func, nu2_func, m12=m12, m21=0)
    # calculate the spectrum.
    fsN1I2 = dadi.Spectrum.from_phi(phiN1I2, (n1,n2), (xx,xx))

    ### Calculate the genomic island spectrum in population 1 and the neutral spectrum in population 2
    # phi for the equilibrium ancestral population
    phiI1N2 = dadi.PhiManip.phi_1D(xx)
    # Now do the divergence event
    phiI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phiI1N2)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tsc, nu1a, nu2a, m12=0, m21=m21)
    # We set the population sizes after the split to nu1 and nu2 and the migration rate to zero
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Ts, nu1a, nu2a, m12=0, m21=0)
    # We keep the population sizes after the split to nu1 and nu2 and set the migration rates to 0 and m21
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Tsc, nu1a, nu2a, m12=0, m21=m21) 
    # population growth
    nu1_func = lambda t: numpy.exp(numpy.log(nu1) * t/Te)
    nu2_func = lambda t: numpy.exp(numpy.log(nu2) * t/Te)
    phiI1N2 = dadi.Integration.two_pops(phiI1N2, xx, Te, nu1_func, nu2_func, m12=0, m21=m21)
    # calculate the spectrum.
    fsI1N2 = dadi.Spectrum.from_phi(phiI1N2, (n1,n2), (xx,xx))

    ### Sum the four spectra
    fs = P1*P2*fsN1N2 + (1-P1)*(1-P2)*fsI1I2 + P1*(1-P2)*fsN1I2 + (1-P1)*P2*fsI1N2
    return fs

def AM2M2P2N(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Tam, Ts, P1, P2, nr, bf = params
    """
    Model with split, ancient migration; two categories of population size and migration rate in the genome.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1 in non-barrier regions.
    m21: Migration from population 1 to population 2 in non-barrier regions.
    Tam: Time of ancient migration.
    Ts: Time of divergence in strict isolation.
    nr: Proportion of "non-recombining" regions affected by background selection.
    bf : Background factor, which defines the extent of population size reduction in "nr" regions.
    P1: Proportion of "non-barrier" regions in population 1.
    P2: Proportion of "non-barrier" regions in population 2.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    # Spectrum for non-recombining regions
    # Spectrum for non-barrier regions in population 1 and 2
    phinrN1N2 = dadi.PhiManip.phi_1D(xx)
    phinrN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrN1N2)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Tam, nu1*bf, nu2*bf, m12=m12, m21=m21)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrN1N2 = dadi.Spectrum.from_phi(phinrN1N2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and 2
    phinrI1I2 = dadi.PhiManip.phi_1D(xx)
    phinrI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrI1I2)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Tam, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrI1I2 = dadi.Spectrum.from_phi(phinrI1I2, (n1,n2), (xx,xx))
    
    # Spectrum for non-barrier regions in population 1 and barrier regions in population 2
    phinrN1I2 = dadi.PhiManip.phi_1D(xx)
    phinrN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrN1I2)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Tam, nu1*bf, nu2*bf, m12=m12, m21=0)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrN1I2 = dadi.Spectrum.from_phi(phinrN1I2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and non-barrier regions in population 2
    phinrI1N2 = dadi.PhiManip.phi_1D(xx)
    phinrI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrI1N2)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Tam, nu1*bf, nu2*bf, m12=0, m21=m21)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrI1N2 = dadi.Spectrum.from_phi(phinrI1N2, (n1,n2), (xx,xx))
    
    # Spectrum for recombining regions
    # Spectrum for non-barrier regions in population 1 and 2
    phirN1N2 = dadi.PhiManip.phi_1D(xx)
    phirN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phirN1N2)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Tam, nu1, nu2, m12=m12, m21=m21)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    fsrN1N2 = dadi.Spectrum.from_phi(phirN1N2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and 2
    phirI1I2 = dadi.PhiManip.phi_1D(xx)
    phirI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phirI1I2)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Tam, nu1, nu2, m12=0, m21=0)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    fsrI1I2 = dadi.Spectrum.from_phi(phirI1I2, (n1,n2), (xx,xx))
    
    # Spectrum for non-barrier regions in population 1 and barrier regions in population 2
    phirN1I2 = dadi.PhiManip.phi_1D(xx)
    phirN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phirN1I2)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Tam, nu1, nu2, m12=m12, m21=0)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    fsrN1I2 = dadi.Spectrum.from_phi(phirN1I2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and non-barrier regions in population 2
    phirI1N2 = dadi.PhiManip.phi_1D(xx)
    phirI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phirI1N2)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Tam, nu1, nu2, m12=0, m21=m21)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    fsrI1N2 = dadi.Spectrum.from_phi(phirI1N2, (n1,n2), (xx,xx))
    fs = nr*(P1*P2*fsnrN1N2 + (1-P1)*(1-P2)*fsnrI1I2 + P1*(1-P2)*fsnrN1I2 + (1-P1)*P2*fsnrI1N2) + (1-nr)*(P1*P2*fsrN1N2 + (1-P1)*(1-P2)*fsrI1I2 + P1*(1-P2)*fsrN1I2 + (1-P1)*P2*fsrI1N2)
    return fs

def IM2M2P2N(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, P1, P2, nr, bf = params

    """
    Model with continuous migration; two categories of population size and migration rate in the genome. 
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1 in non-barrier regions.
    m21: Migration from population 1 to population 2 in non-barrier regions.
    Ts: Time of divergence in continuous migration.
    nr: Proportion of "non-recombining" regions affected by background selection.
    bf : Background factor, which defines the extent of population size reduction in "nr" regions.
    P1: Proportion of "non-barrier" regions in population 1.
    P2: Proportion of "non-barrier" regions in population 2.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    
    xx = dadi.Numerics.default_grid(pts)

    ## Spectrum for non-recombining regions
    # Spectrum for non-barrier regions in population 1 and 2
    phinrN1N2 = dadi.PhiManip.phi_1D(xx)
    phinrN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrN1N2)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Ts, nu1*bf, nu2*bf, m12=m12, m21=m21)
    fsnrN1N2 = dadi.Spectrum.from_phi(phinrN1N2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and 2
    phinrI1I2 = dadi.PhiManip.phi_1D(xx)
    phinrI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrI1I2)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrI1I2 = dadi.Spectrum.from_phi(phinrI1I2, (n1,n2), (xx,xx))

    # Spectrum for non-barrier regions in population 1 and barrier regions in population 2
    phinrN1I2 = dadi.PhiManip.phi_1D(xx)
    phinrN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrN1I2)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Ts, nu1*bf, nu2*bf, m12=m12, m21=0)
    fsnrN1I2 = dadi.Spectrum.from_phi(phinrN1I2, (n1,n2), (xx,xx))

    # Spectrum for barrier regions in population 1 and non-barrier regions in population 2
    phinrI1N2 = dadi.PhiManip.phi_1D(xx)
    phinrI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrI1N2)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=m21)
    fsnrI1N2 = dadi.Spectrum.from_phi(phinrI1N2, (n1,n2), (xx,xx))

    # Spectrum for recombining regions
    # Spectrum for non-barrier regions in population 1 and 2
    phirN1N2 = dadi.PhiManip.phi_1D(xx)
    phirN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phirN1N2)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Ts, nu1, nu2, m12=m12, m21=m21)
    fsrN1N2 = dadi.Spectrum.from_phi(phirN1N2, (n1,n2), (xx,xx))

    # Spectrum for barrier regions in population 1 and 2
    phirI1I2 = dadi.PhiManip.phi_1D(xx)
    phirI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phirI1I2)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    fsrI1I2 = dadi.Spectrum.from_phi(phirI1I2, (n1,n2), (xx,xx))

    # Spectrum for non-barrier regions in population 1 and barrier regions in population 2
    phirN1I2 = dadi.PhiManip.phi_1D(xx)
    phirN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phirN1I2)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Ts, nu1, nu2, m12=m12, m21=0)
    fsrN1I2 = dadi.Spectrum.from_phi(phirN1I2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and non-barrier regions in population 2
    phirI1N2 = dadi.PhiManip.phi_1D(xx)
    phirI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phirI1N2)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Ts, nu1, nu2, m12=0, m21=m21)
    fsrI1N2 = dadi.Spectrum.from_phi(phirI1N2, (n1,n2), (xx,xx))

    fs = nr*(P1*P2*fsnrN1N2 + (1-P1)*(1-P2)*fsnrI1I2 + P1*(1-P2)*fsnrN1I2 + (1-P1)*P2*fsnrI1N2) + (1-nr)*(P1*P2*fsrN1N2 + (1-P1)*(1-P2)*fsrI1I2 + P1*(1-P2)*fsrN1I2 + (1-P1)*P2*fsrI1N2)
    return fs

def PAM2M2P2N(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Tam, Ts, P1, P2, nr, bf = params
    """
    Model with split, two periods of ancient migration; two categories of population size and migration rate in the genome.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1 in non-barrier regions.
    m21: Migration from population 1 to population 2 in non-barrier regions.
    Tam: Time of ancient migration.
    Ts: Time of divergence in strict isolation.
    nr: Proportion of "non-recombining" regions affected by background selection.
    bf : Background factor, which defines the extent of population size reduction in "nr" regions.
    P1: Proportion of "non-barrier" regions in population 1.
    P2: Proportion of "non-barrier" regions in population 2.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    
    # Spectrum for non-recombining regions
    # Spectrum for non-barrier regions in population 1 and 2
    phinrN1N2 = dadi.PhiManip.phi_1D(xx)
    phinrN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrN1N2)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Tam, nu1*bf, nu2*bf, m12=m12, m21=m21)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Tam, nu1*bf, nu2*bf, m12=m12, m21=m21)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrN1N2 = dadi.Spectrum.from_phi(phinrN1N2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and 2
    phinrI1I2 = dadi.PhiManip.phi_1D(xx)
    phinrI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrI1I2)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Tam, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Tam, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrI1I2 = dadi.Spectrum.from_phi(phinrI1I2, (n1,n2), (xx,xx))
    
    # Spectrum for non-barrier regions in population 1 and barrier regions in population 2
    phinrN1I2 = dadi.PhiManip.phi_1D(xx)
    phinrN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrN1I2)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Tam, nu1*bf, nu2*bf, m12=m12, m21=0)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Tam, nu1*bf, nu2*bf, m12=m12, m21=0)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrN1I2 = dadi.Spectrum.from_phi(phinrN1I2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and non-barrier regions in population 2
    phinrI1N2 = dadi.PhiManip.phi_1D(xx)
    phinrI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrI1N2)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Tam, nu1*bf, nu2*bf, m12=0, m21=m21)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Tam, nu1*bf, nu2*bf, m12=0, m21=m21)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrI1N2 = dadi.Spectrum.from_phi(phinrI1N2, (n1,n2), (xx,xx))
    
    # Spectrum for recombining regions
    # Spectrum for non-barrier regions in population 1 and 2
    phirN1N2 = dadi.PhiManip.phi_1D(xx)
    phirN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phirN1N2)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Tam, nu1, nu2, m12=m12, m21=m21)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Tam, nu1, nu2, m12=m12, m21=m21)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    fsrN1N2 = dadi.Spectrum.from_phi(phirN1N2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and 2
    phirI1I2 = dadi.PhiManip.phi_1D(xx)
    phirI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phirI1I2)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Tam, nu1, nu2, m12=0, m21=0)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Tam, nu1, nu2, m12=0, m21=0)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    fsrI1I2 = dadi.Spectrum.from_phi(phirI1I2, (n1,n2), (xx,xx))
    
    # Spectrum for non-barrier regions in population 1 and barrier regions in population 2
    phirN1I2 = dadi.PhiManip.phi_1D(xx)
    phirN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phirN1I2)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Tam, nu1, nu2, m12=m12, m21=0)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Tam, nu1, nu2, m12=m12, m21=0)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    fsrN1I2 = dadi.Spectrum.from_phi(phirN1I2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and non-barrier regions in population 2
    phirI1N2 = dadi.PhiManip.phi_1D(xx)
    phirI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phirI1N2)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Tam, nu1, nu2, m12=0, m21=m21)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Tam, nu1, nu2, m12=0, m21=m21)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    fsrI1N2 = dadi.Spectrum.from_phi(phirI1N2, (n1,n2), (xx,xx))
    
    fs = nr*(P1*P2*fsnrN1N2 + (1-P1)*(1-P2)*fsnrI1I2 + P1*(1-P2)*fsnrN1I2 + (1-P1)*P2*fsnrI1N2) + (1-nr)*(P1*P2*fsrN1N2 + (1-P1)*(1-P2)*fsrI1I2 + P1*(1-P2)*fsrN1I2 + (1-P1)*P2*fsrI1N2)
    return fs

def PSC2M2P2N(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tsc, P1, P2, nr, bf = params
    """
    Model with split, strict isolation, and two periods of secondary contact; two categories of population size and migration rate in the genome.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1 in non-barrier regions.
    m21: Migration from population 1 to population 2 in non-barrier regions.
    Ts: Time of divergence in strict isolation.
    Tsc: Time of secondary contact.
    nr: Proportion of "non-recombining" regions affected by background selection.
    bf : Background factor, which defines the extent of population size reduction in "nr" regions.
    P1: Proportion of "non-barrier" regions in population 1.
    P2: Proportion of "non-barrier" regions in population 2.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    # Spectrum for non-recombining regions
    # Spectrum for non-barrier regions in population 1 and 2
    phinrN1N2 = dadi.PhiManip.phi_1D(xx)
    phinrN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrN1N2)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Tsc, nu1*bf, nu2*bf, m12=m12, m21=m21)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Tsc, nu1*bf, nu2*bf, m12=m12, m21=m21)
    fsnrN1N2 = dadi.Spectrum.from_phi(phinrN1N2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and 2
    phinrI1I2 = dadi.PhiManip.phi_1D(xx)
    phinrI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrI1I2)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Tsc, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Tsc, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrI1I2 = dadi.Spectrum.from_phi(phinrI1I2, (n1,n2), (xx,xx))
    
    # Spectrum for non-barrier regions in population 1 and barrier regions in population 2
    phinrN1I2 = dadi.PhiManip.phi_1D(xx)
    phinrN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrN1I2)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Tsc, nu1*bf, nu2*bf, m12=m12, m21=0)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Tsc, nu1*bf, nu2*bf, m12=m12, m21=0)
    fsnrN1I2 = dadi.Spectrum.from_phi(phinrN1I2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and non-barrier regions in population 2
    phinrI1N2 = dadi.PhiManip.phi_1D(xx)
    phinrI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrI1N2)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Tsc, nu1*bf, nu2*bf, m12=0, m21=m21)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Tsc, nu1*bf, nu2*bf, m12=0, m21=m21)
    fsnrI1N2 = dadi.Spectrum.from_phi(phinrI1N2, (n1,n2), (xx,xx))
    
    # Spectrum for recombining regions
    # Spectrum for non-barrier regions in population 1 and 2
    phirN1N2 = dadi.PhiManip.phi_1D(xx)
    phirN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phirN1N2)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    fsrN1N2 = dadi.Spectrum.from_phi(phirN1N2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and 2
    phirI1I2 = dadi.PhiManip.phi_1D(xx)
    phirI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phirI1I2)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Tsc, nu1, nu2, m12=0, m21=0)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Tsc, nu1, nu2, m12=0, m21=0)
    fsrI1I2 = dadi.Spectrum.from_phi(phirI1I2, (n1,n2), (xx,xx))
    
    # Spectrum for non-barrier regions in population 1 and barrier regions in population 2
    phirN1I2 = dadi.PhiManip.phi_1D(xx)
    phirN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phirN1I2)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Tsc, nu1, nu2, m12=m12, m21=0)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Tsc, nu1, nu2, m12=m12, m21=0)
    fsrN1I2 = dadi.Spectrum.from_phi(phirN1I2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and non-barrier regions in population 2
    phirI1N2 = dadi.PhiManip.phi_1D(xx)
    phirI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phirI1N2)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Tsc, nu1, nu2, m12=0, m21=m21)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Tsc, nu1, nu2, m12=0, m21=m21)
    fsrI1N2 = dadi.Spectrum.from_phi(phirI1N2, (n1,n2), (xx,xx))
    fs = nr*(P1*P2*fsnrN1N2 + (1-P1)*(1-P2)*fsnrI1I2 + P1*(1-P2)*fsnrN1I2 + (1-P1)*P2*fsnrI1N2) + (1-nr)*(P1*P2*fsrN1N2 + (1-P1)*(1-P2)*fsrI1I2 + P1*(1-P2)*fsrN1I2 + (1-P1)*P2*fsrI1N2)

    return fs

def SC2M2P2N(params, (n1,n2), pts):
    nu1, nu2, m12, m21, Ts, Tsc, P1, P2, nr, bf = params
    """
    Model with split, strict isolation, and secondary contact; two categories of population size and migration rate in the genome.
    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration from population 2 to population 1 in non-barrier regions.
    m21: Migration from population 1 to population 2 in non-barrier regions.
    Ts: Time of divergence in strict isolation.
    Tsc: Time of secondary contact.
    nr: Proportion of "non-recombining" regions affected by background selection.
    bf : Background factor, which defines the extent of population size reduction in "nr" regions.
    P1: Proportion of "non-barrier" regions in population 1.
    P2: Proportion of "non-barrier" regions in population 2.
    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    xx = dadi.Numerics.default_grid(pts)
    # Spectrum for non-recombining regions
    # Spectrum for non-barrier regions in population 1 and 2
    phinrN1N2 = dadi.PhiManip.phi_1D(xx)
    phinrN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrN1N2)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrN1N2 = dadi.Integration.two_pops(phinrN1N2, xx, Tsc, nu1*bf, nu2*bf, m12=m12, m21=m21)
    fsnrN1N2 = dadi.Spectrum.from_phi(phinrN1N2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and 2
    phinrI1I2 = dadi.PhiManip.phi_1D(xx)
    phinrI1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrI1I2)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1I2 = dadi.Integration.two_pops(phinrI1I2, xx, Tsc, nu1*bf, nu2*bf, m12=0, m21=0)
    fsnrI1I2 = dadi.Spectrum.from_phi(phinrI1I2, (n1,n2), (xx,xx))
    
    # Spectrum for non-barrier regions in population 1 and barrier regions in population 2
    phinrN1I2 = dadi.PhiManip.phi_1D(xx)
    phinrN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrN1I2)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrN1I2 = dadi.Integration.two_pops(phinrN1I2, xx, Tsc, nu1*bf, nu2*bf, m12=m12, m21=0)
    fsnrN1I2 = dadi.Spectrum.from_phi(phinrN1I2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and non-barrier regions in population 2
    phinrI1N2 = dadi.PhiManip.phi_1D(xx)
    phinrI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phinrI1N2)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Ts, nu1*bf, nu2*bf, m12=0, m21=0)
    phinrI1N2 = dadi.Integration.two_pops(phinrI1N2, xx, Tsc, nu1*bf, nu2*bf, m12=0, m21=m21)
    fsnrI1N2 = dadi.Spectrum.from_phi(phinrI1N2, (n1,n2), (xx,xx))
    
    # Spectrum for recombining regions
    # Spectrum for non-barrier regions in population 1 and 2
    phirN1N2 = dadi.PhiManip.phi_1D(xx)
    phirN1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phirN1N2)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirN1N2 = dadi.Integration.two_pops(phirN1N2, xx, Tsc, nu1, nu2, m12=m12, m21=m21)
    fsrN1N2 = dadi.Spectrum.from_phi(phirN1N2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and 2
    phirI1I2 = dadi.Integration.two_pops(phirI1I2, xx, Tsc, nu1, nu2, m12=0, m21=0)
    fsrI1I2 = dadi.Spectrum.from_phi(phirI1I2, (n1,n2), (xx,xx))
    
    # Spectrum for non-barrier regions in population 1 and barrier regions in population 2
    phirN1I2 = dadi.PhiManip.phi_1D(xx)
    phirN1I2 = dadi.PhiManip.phi_1D_to_2D(xx, phirN1I2)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirN1I2 = dadi.Integration.two_pops(phirN1I2, xx, Tsc, nu1, nu2, m12=m12, m21=0)
    fsrN1I2 = dadi.Spectrum.from_phi(phirN1I2, (n1,n2), (xx,xx))
    
    # Spectrum for barrier regions in population 1 and non-barrier regions in population 2
    phirI1N2 = dadi.PhiManip.phi_1D(xx)
    phirI1N2 = dadi.PhiManip.phi_1D_to_2D(xx, phirI1N2)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Ts, nu1, nu2, m12=0, m21=0)
    phirI1N2 = dadi.Integration.two_pops(phirI1N2, xx, Tsc, nu1, nu2, m12=0, m21=m21)
    fsrI1N2 = dadi.Spectrum.from_phi(phirI1N2, (n1,n2), (xx,xx))
    fs = nr*(P1*P2*fsnrN1N2 + (1-P1)*(1-P2)*fsnrI1I2 + P1*(1-P2)*fsnrN1I2 + (1-P1)*P2*fsnrI1N2) + (1-nr)*(P1*P2*fsrN1N2 + (1-P1)*(1-P2)*fsrI1I2 + P1*(1-P2)*fsrN1I2 + (1-P1)*P2*fsrI1N2)

    return fs


