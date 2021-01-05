from seirsplus.models import *
from seirsplus.networks import *
from seirsplus.sim_loops import *
from seirsplus.utilities import *
import modelVac
import networkx
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import csv

def run_model(scenario,runs,plot,outfile):
    #Total population size
    N = 10800

    #Initial infected number
    INIT_INFECTED = 1

    #Specifying contact network

    #Using demographic community network generator defined in the SEIRS+ package
    household_data = {
                       'age_distn':{'0-9': 0.09706, '10-19':0.09917, '20-29': 0.12489, '30-39': 0.15119, '40-49': 0.14618, '50-59': 0.12868, '60-69': 0.10485, '70-79': 0.09405, '80+'  : 0.05393 },
                       'household_size_distn':{ 1: 0.2567, 2: 0.2947, 3: 0.1978, 4: 0.1757, 5: 0.05069, 6: 0.0167, 7: 0.00771 },
                       'household_stats':{ 'pct_with_under20': 0.28,                      # percent of households with at least one member under 60
                                           'pct_with_over60': 0.42,                       # percent of households with at least one member over 60
                                           'pct_with_under20_over60':  0.05,              # percent of households with at least one member under 20 and at least one member over 60
                                           'pct_with_over60_givenSingleOccupant': 0.12,   # percent of households with a single-occupant that is over 60
                                           'mean_num_under20_givenAtLeastOneUnder20': 1.62 # number of people under 20 in households with at least one member under 20
                                         }
                     }

    layer_info  = { '0-9':   {'ageBrackets': ['0-9'],   'meanDegree': 12.62 },
                    '10-19': {'ageBrackets': ['10-19'], 'meanDegree': 15.38 },
                    '20-29': {'ageBrackets': ['20-29'], 'meanDegree': 12.89 },
                    '30-39':   {'ageBrackets': ['30-39'], 'meanDegree': 12.89},
                    '40-49':   {'ageBrackets': ['40-49'], 'meanDegree': 12.27},
                    '50-59':   {'ageBrackets': ['50-59'], 'meanDegree': 11.64},
                    '60-69':   {'ageBrackets': ['60-69'], 'meanDegree': 9.42},
                    '70-79':   {'ageBrackets': ['70-79'], 'meanDegree': 7.2},
                    '80+':   {'ageBrackets': ['80+'], 'meanDegree': 7.2} }

    demographic_graphs, individual_ageGroups, households = generate_demographic_contact_network(
                                                                N=N ,demographic_data=household_data,layer_generator='FARZ',layer_info=layer_info)

    G_baseline   = demographic_graphs['baseline']

    households_indices = [household['indices'] for household in households]

    #Specifying parameters

    SIGMA = 1/3.5

    LAMDA = 1/1.5

    #----------------------

    symptomaticPeriod_mean, symptomaticPeriod_coeffvar = 6.0, 0.4
    GAMMA   = 1 / gamma_dist(symptomaticPeriod_mean, symptomaticPeriod_coeffvar, N)

    infectiousPeriod = 1/LAMDA + 1/GAMMA

    GAMMA_asym = GAMMA

    #----------------------

    onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar = 5.0, 0.2
    ETA     = 1 / gamma_dist(onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar, N)

    hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar = 11.5, 0.2
    GAMMA_H = 1 / gamma_dist(hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar, N)

    #----------------------

    hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar = 11.774, 0.7487
    MU_H    = 1 / gamma_dist(hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar, N)

    #Severity parameters

    #Cases that are asymptomatic
    ageGroup_pctAsymptomatic = {'0-9':      0.456,
                                '10-19':    0.412,
                                '20-29':    0.37,
                                '30-39':    0.332,
                                '40-49':    0.296,
                                '50-59':    0.265,
                                '60-69':    0.238,
                                '70-79':    0.214,
                                '80+':      0.192}

    PCT_ASYMPTOMATIC = [ageGroup_pctAsymptomatic[ageGroup] for ageGroup in individual_ageGroups]

    #age-stratified case hospitalization rates
    ageGroup_pctHospitalized = {'0-9':      0.000011,
                                '10-19':    0.000114,
                                '20-29':    0.000495,
                                '30-39':    0.003726,
                                '40-49':    0.019272,
                                '50-59':    0.037107,
                                '60-69':    0.07067,
                                '70-79':    0.102833,
                                '80+':      0.131513 }
    PCT_HOSPITALIZED = [ageGroup_pctHospitalized[ageGroup] for ageGroup in individual_ageGroups]

    #fatality rates for hospitalized cases
    ageGroup_hospitalFatalityRate = {'0-9':     0.0165,
                                     '10-19':   0.0125,
                                     '20-29':   0.025,
                                     '30-39':   0.025,
                                     '40-49':   0.0315,
                                     '50-59':   0.08418,
                                     '60-69':   0.1781,
                                     '70-79':   0.38016,
                                     '80+':     0.709}
    PCT_FATALITY = [ageGroup_hospitalFatalityRate[ageGroup] for ageGroup in individual_ageGroups]

    #Set transmission parameters

    #the expected number of new infections generated by a single particular infectious individual in a completely susceptible population
    R0_mean     = 2.38
    R0_coeffvar = 0.3

    R0 = gamma_dist(R0_mean, R0_coeffvar, N)

    #The means of the Individual Transmissibility Values for infectious subpopulations
    BETA = 0.155
    BETA_asym = BETA/2

    #which define the local transmissibility for each pair of close contacts
    BETA_PAIRWISE_MODE  = 'infected'

    DELTA_PAIRWISE_MODE = 'mean'

    #Here we set individual susceptibilities for different age groups(default susceptibility is 1).
    #We set p to reflect 20% of interactions being with incidental or casual contacts outside their set of close contacts for different age groups.

    for age in individual_ageGroups:
        if age == '0-9':
            ALPHA = 0.35
            P_GLOBALINTXN = 0.9
        elif age == '10-19':
            ALPHA = 0.69
            P_GLOBALINTXN = 0.95
        elif age == '20-29':
            ALPHA = 1.03
            P_GLOBALINTXN = 0.7
        elif age == '30-39':
            ALPHA = 1.03
            P_GLOBALINTXN = 0.9
        elif age == '40-49':
            ALPHA = 1.03
            P_GLOBALINTXN = 0.9
        elif age == '50-59':
            ALPHA = 1.03
            P_GLOBALINTXN = 0.9
        elif age == '60-69':
            ALPHA = 1.27
            P_GLOBALINTXN = 0.6
        elif age == '70-79':
            ALPHA = 1.52
            P_GLOBALINTXN = 0.5
        elif age == '80+':
            ALPHA = 1.52
            P_GLOBALINTXN = 0.5

    #Define p changes for each measure

    #School and University closures
    ageGroup_p_SchoolUniClosure = {'0-9':   0.5,
                                '10-19':    0.5,
                                '20-29':    0.6,
                                '30-39':    0.6,
                                '40-49':    0.3,
                                '50-59':    0.3,
                                '60-69':    0.3,
                                '70-79':    0.2,
                                '80+':      0.2}
    p_SchoolUniClosure = [ageGroup_p_SchoolUniClosure[ageGroup] for ageGroup in individual_ageGroups]

    #Leisure closure
    ageGroup_p_LeisureClosure = {'0-9':   0.3,
                                '10-19':    0.3,
                                '20-29':    0.4,
                                '30-39':    0.4,
                                '40-49':    0.3,
                                '50-59':    0.3,
                                '60-69':    0.3,
                                '70-79':    0.2,
                                '80+':      0.2}
    p_LeisureClosure = [ageGroup_p_SchoolUniClosure[ageGroup] for ageGroup in individual_ageGroups]

    #1st National Lockdown
    ageGroup_p_National_Lockdown1 = {'0-9':   0.02,
                                '10-19':    0.02,
                                '20-29':    0.05,
                                '30-39':    0.02,
                                '40-49':    0.02,
                                '50-59':    0.02,
                                '60-69':    0.02,
                                '70-79':    0.02,
                                '80+':      0.02}
    p_NationalLockdown = [ageGroup_p_SchoolUniClosure[ageGroup] for ageGroup in individual_ageGroups]

    #Leisure – mass gathering containment
    ageGroup_p_LeisureMass = {'0-9':   0.6,
                                '10-19':    0.6,
                                '20-29':    0.7,
                                '30-39':    0.7,
                                '40-49':    0.5,
                                '50-59':    0.5,
                                '60-69':    0.5,
                                '70-79':    0.3,
                                '80+':      0.3}
    p_LeisureMass = [ageGroup_p_LeisureMass[ageGroup] for ageGroup in individual_ageGroups]

    #Leisure 24/10
    ageGroup_p_Leisure24_10 = {'0-9':   0.9,
                                '10-19':    0.5,
                                '20-29':    0.6,
                                '30-39':    0.6,
                                '40-49':    0.4,
                                '50-59':    0.4,
                                '60-69':    0.4,
                                '70-79':    0.2,
                                '80+':      0.2}
    p_Leisure24_10 = [ageGroup_p_Leisure24_10[ageGroup] for ageGroup in individual_ageGroups]



    #2nd national lockdown
    ageGroup_p_National_Lockdown2 = {'0-9':   0.3,
                                '10-19':    0.3,
                                '20-29':    0.1,
                                '30-39':    0.03,
                                '40-49':    0.03,
                                '50-59':    0.03,
                                '60-69':    0.03,
                                '70-79':    0.03,
                                '80+':      0.03}
    p_NationalLockdown2 = [ageGroup_p_National_Lockdown2[ageGroup] for ageGroup in individual_ageGroups]

    #EducationClosure
    ageGroup_p_EducationClosure = {'0-9':   0.03,
                                '10-19':    0.03,
                                '20-29':    0.1,
                                '30-39':    0.03,
                                '40-49':    0.03,
                                '50-59':    0.03,
                                '60-69':    0.03,
                                '70-79':    0.03,
                                '80+':      0.03}
    p_EducationClosure = [ageGroup_p_National_Lockdown2[ageGroup] for ageGroup in individual_ageGroups]


    #Define BETA changes for each measure

    #“Light” masking
    ageGroup_BETA_LightMasking = {'0-9':   0.95*BETA,
                                '10-19':    0.95*BETA,
                                '20-29':    0.95*BETA,
                                '30-39':    0.95*BETA,
                                '40-49':    0.95*BETA,
                                '50-59':    0.8*BETA,
                                '60-69':    0.8*BETA,
                                '70-79':    0.8*BETA,
                                '80+':      0.8*BETA}
    BETA_LightMasking = [ageGroup_BETA_LightMasking[ageGroup] for ageGroup in individual_ageGroups]

    ageGroup_BETA_asym_LightMasking = {'0-9':   0.95*BETA,
                                '10-19':    0.95*BETA,
                                '20-29':    0.95*BETA,
                                '30-39':    0.95*BETA,
                                '40-49':    0.95*BETA,
                                '50-59':    0.8*BETA,
                                '60-69':    0.8*BETA,
                                '70-79':    0.8*BETA,
                                '80+':      0.8*BETA}
    BETA__asym_LightMasking = [ageGroup_BETA_LightMasking[ageGroup] for ageGroup in individual_ageGroups]

    #"Heavy” masking +teleworking 50%
    ageGroup_BETA_HeavyMaskingTeleworking = {'0-9':   0.5*BETA,
                                            '10-19':    0.5*BETA,
                                            '20-29':    0.65*BETA,
                                            '30-39':    0.65*BETA,
                                            '40-49':    0.65*BETA,
                                            '50-59':    0.65*BETA,
                                            '60-69':    0.2*BETA,
                                            '70-79':    0.2*BETA,
                                            '80+':      0.2*BETA}
    BETA_HeavyMaskingTeleworking = [ageGroup_BETA_HeavyMaskingTeleworking[ageGroup] for ageGroup in individual_ageGroups]

    #"Heavy” masking +teleworking 50%
    ageGroup_BETA_asym_HeavyMaskingTeleworking = {'0-9':   0.5*BETA_asym,
                                            '10-19':    0.5*BETA_asym,
                                            '20-29':    0.65*BETA_asym,
                                            '30-39':    0.65*BETA_asym,
                                            '40-49':    0.65*BETA_asym,
                                            '50-59':    0.65*BETA_asym,
                                            '60-69':    0.2*BETA_asym,
                                            '70-79':    0.2*BETA_asym,
                                            '80+':      0.2*BETA_asym}
    BETA_asym_HeavyMaskingTeleworking = [ageGroup_BETA_asym_HeavyMaskingTeleworking[ageGroup] for ageGroup in individual_ageGroups]

    #Defining different networks for various measures

    # Graph for Schools and University closures
    ageGroup_scale_SchoolUniClosure = {'0-9':   15,
                                        '10-19':    74,
                                        '20-29':    57,
                                        '30-39':    74,
                                        '40-49':    74,
                                        '50-59':    57,
                                        '60-69':    33,
                                        '70-79':    20,
                                        '80+':      20}
    scale_SchoolUniClosure = [ageGroup_scale_SchoolUniClosure[ageGroup] for ageGroup in individual_ageGroups]
    ageGroup_m_SchoolUniClosure = {'0-9':   6,
                                        '10-19':    12,
                                        '20-29':    11,
                                        '30-39':    12,
                                        '40-49':    12,
                                        '50-59':    11,
                                        '60-69':    9,
                                        '70-79':    7,
                                        '80+':      7}
    m_SchoolUniClosure = [ageGroup_m_SchoolUniClosure[ageGroup] for ageGroup in individual_ageGroups]
    G_SchoolUniClosure = custom_exponential_graph( G_baseline, scale = scale_SchoolUniClosure,m = m_SchoolUniClosure)

    #Graph for Leisure closure
    ageGroup_scale_LeisureClosure = {'0-9':   7,
                                        '10-19':    7,
                                        '20-29':    20,
                                        '30-39':    15,
                                        '40-49':    15,
                                        '50-59':    15,
                                        '60-69':    25,
                                        '70-79':    20,
                                        '80+':      20}
    scale_LeisureClosure = [ageGroup_scale_LeisureClosure[ageGroup] for ageGroup in individual_ageGroups]

    ageGroup_m_LeisureClosure = {'0-9':   3,
                                        '10-19':    3,
                                        '20-29':    7,
                                        '30-39':    6,
                                        '40-49':    6,
                                        '50-59':    6,
                                        '60-69':    8,
                                        '70-79':    7,
                                        '80+':      7}
    m_LeisureClosure = [ageGroup_m_LeisureClosure[ageGroup] for ageGroup in individual_ageGroups]
    G_LeisureClosure = custom_exponential_graph( G_baseline, scale = scale_LeisureClosure,m=m_LeisureClosure )

    #Private enterprises closure
    ageGroup_scale_PrivateEnterprises = {'0-9':   7,
                                        '10-19':    7,
                                        '20-29':    15,
                                        '30-39':    11,
                                        '40-49':    11,
                                        '50-59':    11,
                                        '60-69':    15,
                                        '70-79':    10,
                                        '80+':      10}
    scale_PrivateEnterprises = [ageGroup_scale_PrivateEnterprises[ageGroup] for ageGroup in individual_ageGroups]
    ageGroup_m_PrivateEnterprises = {'0-9':   3,
                                        '10-19':    3,
                                        '20-29':    6,
                                        '30-39':    5,
                                        '40-49':    5,
                                        '50-59':    5,
                                        '60-69':    6,
                                        '70-79':    4,
                                        '80+':      4}
    m_PrivateEnterprises = [ageGroup_m_PrivateEnterprises[ageGroup] for ageGroup in individual_ageGroups]
    G_PrivateEnterprisesClosure = custom_exponential_graph( G_baseline, scale = scale_PrivateEnterprises)

    #1st National Lockdown
    ageGroup_scale_1stLockdown = {'0-9':   7,
                                        '10-19':    7,
                                        '20-29':    11,
                                        '30-39':    7,
                                        '40-49':    7,
                                        '50-59':    7,
                                        '60-69':    7,
                                        '70-79':    5,
                                        '80+':      5}
    scale_1stLockdown = [ageGroup_scale_1stLockdown[ageGroup] for ageGroup in individual_ageGroups]
    ageGroup_m_1stLockdown = {'0-9':   3,
                                        '10-19':    3,
                                        '20-29':    5,
                                        '30-39':    3,
                                        '40-49':    3,
                                        '50-59':    3,
                                        '60-69':    3,
                                        '70-79':    2,
                                        '80+':      2}
    m_1stLockdown = [ageGroup_m_1stLockdown[ageGroup] for ageGroup in individual_ageGroups]
    G_1stLockdown = custom_exponential_graph( G_baseline, scale = scale_1stLockdown,m=m_1stLockdown)

    #Graph for Leisure – mass gathering containment
    ageGroup_scale_LeisureMass = {'0-9':   100,
                                        '10-19':    100,
                                        '20-29':    100,
                                        '30-39':    74,
                                        '40-49':    74,
                                        '50-59':    57,
                                        '60-69':    33,
                                        '70-79':    20,
                                        '80+':      20}
    scale_LeisureMass = [ageGroup_scale_LeisureMass[ageGroup] for ageGroup in individual_ageGroups]
    ageGroup_m_LeisureMass = {'0-9':   13,
                                        '10-19':    13,
                                        '20-29':    13,
                                        '30-39':    12,
                                        '40-49':    12,
                                        '50-59':    11,
                                        '60-69':    9,
                                        '70-79':    7,
                                        '80+':      7}
    m_LeisureMass = [ageGroup_m_LeisureMass[ageGroup] for ageGroup in individual_ageGroups]
    G_LeisureMass = custom_exponential_graph( G_baseline, scale = scale_LeisureMass,m=m_LeisureMass)

    #Graph for Leisure closure 2
    ageGroup_scale_LeisureClosure2 = {'0-9':   100,
                                        '10-19':    214,
                                        '20-29':    74,
                                        '30-39':    57,
                                        '40-49':    57,
                                        '50-59':    57,
                                        '60-69':    25,
                                        '70-79':    20,
                                        '80+':      20}
    scale_LeisureClosure2 = [ageGroup_scale_LeisureClosure2[ageGroup] for ageGroup in individual_ageGroups]
    ageGroup_m_LeisureClosure2 = {'0-9':   13,
                                        '10-19':    16,
                                        '20-29':    12,
                                        '30-39':    11,
                                        '40-49':    11,
                                        '50-59':    11,
                                        '60-69':    8,
                                        '70-79':    7,
                                        '80+':      7}
    m_LeisureClosure2 = [ageGroup_m_LeisureClosure2[ageGroup] for ageGroup in individual_ageGroups]
    G_baseline_NODIST = demographic_graphs['baseline'].copy()
    G_LeisureClosure2 = custom_exponential_graph( G_baseline, scale = scale_LeisureClosure2,m=m_LeisureClosure2)

    #«Heavy” masking +teleworking 50%
    ageGroup_scale_HeavyMaskingTeleworking = {'0-9':   100,
                                        '10-19':    214,
                                        '20-29':    74,
                                        '30-39':    43,
                                        '40-49':    43,
                                        '50-59':    43,
                                        '60-69':    20,
                                        '70-79':    20,
                                        '80+':      20}
    scale_HeavyMaskingTeleworking = [ageGroup_scale_HeavyMaskingTeleworking[ageGroup] for ageGroup in individual_ageGroups]
    ageGroup_m_HeavyMaskingTeleworking = {'0-9':   13,
                                        '10-19':    16,
                                        '20-29':    12,
                                        '30-39':    10,
                                        '40-49':    10,
                                        '50-59':    10,
                                        '60-69':    7,
                                        '70-79':    7,
                                        '80+':      7}
    m_HeavyMaskingTeleworking = [ageGroup_m_HeavyMaskingTeleworking[ageGroup] for ageGroup in individual_ageGroups]
    G_HeavyMaskingTeleworking = custom_exponential_graph(G_baseline, scale = scale_HeavyMaskingTeleworking,m=m_HeavyMaskingTeleworking)

    #Graph for 2nd National Lockdown
    ageGroup_scale_Lockdown2 = {'0-9':   100,
                                        '10-19':    20,
                                        '20-29':    25,
                                        '30-39':    20,
                                        '40-49':    20,
                                        '50-59':    20,
                                        '60-69':    20,
                                        '70-79':    7,
                                        '80+':      7}
    scale_Lockdown2 = [ageGroup_scale_Lockdown2[ageGroup] for ageGroup in individual_ageGroups]
    ageGroup_m_Lockdown2 = {'0-9':   13,
                                        '10-19':    7,
                                        '20-29':    8,
                                        '30-39':    7,
                                        '40-49':    7,
                                        '50-59':    7,
                                        '60-69':    7,
                                        '70-79':    3,
                                        '80+':      3}
    m_Lockdown2 = [ageGroup_m_Lockdown2[ageGroup] for ageGroup in individual_ageGroups]
    G_Lockdown2 = custom_exponential_graph(G_baseline, scale = scale_Lockdown2,m=m_Lockdown2)

    #Nursery schools – kindergarten – primary schools closure
    ageGroup_scale_EducationClosure = {'0-9':   10,
                                        '10-19':    10,
                                        '20-29':    20,
                                        '30-39':    11,
                                        '40-49':    11,
                                        '50-59':    11,
                                        '60-69':    11,
                                        '70-79':    7,
                                        '80+':      7}
    scale_EducationClosure = [ageGroup_scale_EducationClosure[ageGroup] for ageGroup in individual_ageGroups]
    ageGroup_m_EducationClosure = {'0-9':   4,
                                        '10-19':    4,
                                        '20-29':    7,
                                        '30-39':    5,
                                        '40-49':    5,
                                        '50-59':    5,
                                        '60-69':    5,
                                        '70-79':    3,
                                        '80+':      3}
    m_EducationClosure = [ageGroup_m_EducationClosure[ageGroup] for ageGroup in individual_ageGroups]
    G_EducationClosure = custom_exponential_graph( G_baseline, scale = scale_EducationClosure,m = m_EducationClosure)


    checkpoints = {}
    checkpoints = {'t':      [14,22,21,26,68,75,81,82,89,96,180,216,241,255,262,293,317],
                   'G':       [G_SchoolUniClosure,#11/3 14
                                G_LeisureClosure,#14/3 17
                                G_PrivateEnterprisesClosure,#18/3 21
                                G_1stLockdown,#23/3 26
                                G_PrivateEnterprisesClosure,#4/5  68
                                G_LeisureClosure,#11/5 75
                                G_LeisureClosure,#17/5 81
                                G_LeisureClosure,#18/5 82
                                G_SchoolUniClosure,#25/5 89
                                G_baseline,#1/6 96
                                G_baseline,#24/8 180
                                G_LeisureClosure2,#29/9 216
                                G_HeavyMaskingTeleworking,#24/10 241
                                G_Lockdown2,#7/11 255
                                G_EducationClosure,#14/11 262
                                G_SchoolUniClosure,#G_HeavyMaskingTeleworking,#15/12 293
                                G_HeavyMaskingTeleworking],#8/1 317
                   'p':       [p_SchoolUniClosure,#11/3 14
                               p_LeisureClosure,#14/3 17
                               p_LeisureClosure,#18/3 31
                               p_NationalLockdown,#23/3 26
                               p_LeisureClosure,#4/5  68
                               p_LeisureClosure,#11/5 75
                               p_LeisureClosure,#17/5 81
                               p_SchoolUniClosure,#18/5 82
                               p_SchoolUniClosure,#25/5 89
                               P_GLOBALINTXN,#1/6 96
                               P_GLOBALINTXN,#p_LeisureMass,#24/8 180
                               p_LeisureMass,#p_LeisureClosure,#29/9 216
                               p_Leisure24_10,#p_LeisureClosure,#24/10 241
                               p_NationalLockdown2,#7/11 255
                               p_EducationClosure,#14/11 262
                               p_SchoolUniClosure,#15/12 293
                               p_LeisureMass],#8/1 317
                   'BETA':    [BETA,BETA,BETA,BETA,BETA_LightMasking,BETA_LightMasking,BETA_LightMasking,BETA_LightMasking,BETA_LightMasking,BETA_LightMasking,BETA_LightMasking,BETA_LightMasking,BETA_HeavyMaskingTeleworking,BETA_HeavyMaskingTeleworking,BETA_HeavyMaskingTeleworking,BETA_HeavyMaskingTeleworking,BETA_HeavyMaskingTeleworking],
                   'BETA_asym':[BETA_asym,BETA_asym,BETA_asym,BETA_asym,BETA__asym_LightMasking,BETA__asym_LightMasking,BETA__asym_LightMasking,BETA__asym_LightMasking,BETA__asym_LightMasking,BETA__asym_LightMasking,BETA__asym_LightMasking,BETA__asym_LightMasking,BETA_asym_HeavyMaskingTeleworking,BETA_asym_HeavyMaskingTeleworking,BETA_asym_HeavyMaskingTeleworking,BETA_asym_HeavyMaskingTeleworking,BETA_asym_HeavyMaskingTeleworking]}

    checkpoints2Freedom = {'t':[293],
                          'G': [G_baseline],
                          'p': [P_GLOBALINTXN],
                          'BETA': [BETA]}

    checkpoints2Lockdown ={'t':[293,317,340],
                          'G': [G_SchoolUniClosure,G_HeavyMaskingTeleworking,G_Lockdown2],
                          'p': [p_SchoolUniClosure,p_LeisureMass,p_NationalLockdown2],
                          'BETA': [BETA_HeavyMaskingTeleworking,BETA_HeavyMaskingTeleworking,BETA_HeavyMaskingTeleworking]}

    #Initializing the model

    model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                  beta=BETA,beta_asym=BETA_asym, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                  gamma_asym=GAMMA_asym, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H,
                                  a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,
                                  alpha=ALPHA,beta_pairwise_mode=BETA_PAIRWISE_MODE, delta_pairwise_mode=DELTA_PAIRWISE_MODE, q=0,
                                  initI_pre=INIT_INFECTED)

    #Running the model and introducing confirmed cases

    model.run(T=36,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#23/3 36

    model.run(T=8,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#5/4 44

    model.run(T=58,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#2/6 102

    model.run(T=42,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#14/7 144

    model.run(T=17,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#31/7 161

    model.run(T=6,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#6/8 167

    model.run(T=5,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#11/8 172

    model.run(T=4,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#15/8 176

    model.run(T=5,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#20/8 181

    model.run(T=4,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#24/8 185

    model.run(T=6,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#30/8 191

    model.run(T=5,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#4/9 196

    model.run(T=3,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#7/9 199

    model.run(T=4,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#11/9 203

    model.run(T=3,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#14/9 206

    model.run(T=3,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#17/9 209

    model.run(T=3,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#20/9 212

    model.run(T=4,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#24/9 216

    model.run(T=2,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#26/9 218

    model.run(T=4,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#30/9 222

    model.run(T=2,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#2/10 224

    model.run(T=3,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#5/10 227

    model.run(T=3,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#8/10 230

    model.run(T=2,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#10/10 232

    model.run(T=2,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#12/10 234

    model.run(T=2,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#14/10 236

    model.run(T=2,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#16/10 238

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#17/10 239

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#18/10 240

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#19/10 241

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#20/10 242

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#31/10 243

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#22/10 244

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#23/10 245

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#24/10 246

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#25/10 247

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#26/10 248

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#27/10 249

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#28/10 250

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#29/10 251

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#30/10 252

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=3)#31/10 253

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#1/11 254

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=3)#2/11 255

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#3/11 256

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#4/11 257

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#5/11 258

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=3)#6/11 259

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=3)#7/11 260

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=4)#8/11 261

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=3)#9/11 262

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#10/11 263

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#11/11 264

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#12/11 265

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=3)#13/11 266

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=3)#14/11 267

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=3)#15/11 268

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#16/11 269

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#17/11 270

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#18/11 271

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#19/11 272

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#20/11 273

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#21/11 274

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#22/11 275

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#23/11 276

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#24/11 277

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#25/11 278

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#26/11 279

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#27/11 280

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#28/11 281

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#29/11 282

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#30/11 283

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#1/12 284

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#2/12 285

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#3/12 286

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#4/12 285

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=2)#5/12 286

    model.run(T=1,verbose=True,checkpoints=checkpoints)
    model.introduce_exposures(num_new_exposures=1)#6/12 287

    #Running different scenarios
    if(scenario == 'freedom'):
        model.run(T=81,verbose=True,checkpoints=checkpoints2Freedom)
    elif(scenario == 'semi'):
        model.run(T=81,verbose=True,checkpoints=checkpoints)
    elif(scenario == 'lockdown'):
        model.run(T=81,verbose=True,checkpoints=checkpoints2Lockdown)

    #Plotting number of nodes in each state along with the most important measures taken
    checkpointsToPlot = [14, 17, 21, 26, 68, 216, 241, 255, 262]
    if(plot):model.figure_infections(vlines=checkpointsToPlot,vline_labels = ['11/03 Schools/Uni Closure','13-14/03 Leisure closure',
    '18/03 Private enterprises closure','23/03 1st National Lockdown','04/05 End of lockdown',
    '24/08 Leisure and mass gathering containment','29/09 Leisure closure','24/10 Heavy masking + 50% teleworking'
    ,'07/11 2nd National Lockdown','14/11 Education closure'],plot_percentages=False,
                            ylim=100000,vline_colors=['red','purple','green','black','black','orange','blue','lightblue','pink'],
                            )

    #Getting numbers for each state at the end of the model
    timeSeries   = model.tseries
    Sseries      = model.numS
    Eseries      = model.numE
    I_preseries  = model.numI_pre
    I_symseries  = model.numI_sym
    I_asymseries = model.numI_asym
    Rseries      = model.numR
    Hseries      = model.numH
    Fseries      = model.numF

    if (scenario == 'semi'):
        #Start vaccination model/phase 2
        model2 = modelVac.ExtSEIRSNetworkModelVac(G=G_HeavyMaskingTeleworking, p=p_LeisureMass,
                                      beta=BETA_HeavyMaskingTeleworking,beta_asym=BETA_asym_HeavyMaskingTeleworking, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                      gamma_asym=GAMMA_asym, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H,
                                      a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,
                                      alpha=ALPHA,beta_pairwise_mode=BETA_PAIRWISE_MODE, delta_pairwise_mode=DELTA_PAIRWISE_MODE, q=0,
                                      initR=448,
                                      initE=4, initI_pre=3, initI_sym=6,
                                       initI_asym=3, initF=8,initH=1)
        checkpointsVac = {'t':[45],
                          'G': [G_baseline],
                          'p': [P_GLOBALINTXN],
                          'BETA': [BETA_LightMasking],
                          'BETA_asym':[BETA__asym_LightMasking]}

        model2.run(T=1,verbose=True,checkpoints=checkpointsVac)
        model2.introduce_vaccined(num_new_vaccined=90)

        model2.run(T=134,verbose=True,checkpoints=checkpointsVac)
        if(plot):model2.figure_infections(ylim=60000,plot_percentages=False,vlines=checkpointsVac['t'],vline_labels = ['1/4 Lighter measures'])
        vactimeSeries   = model2.tseries
        vacSseries      = model2.numS
        vacEseries      = model2.numE
        vacI_preseries  = model2.numI_pre
        vacI_symseries  = model2.numI_sym
        vacI_asymseries = model2.numI_asym
        vacRseries      = model2.numR
        vacHseries      = model2.numH
        vacFseries      = model2.numF

    elif(scenario == 'lockdown'):
        #Start vaccination model/phase 2
        model2 = modelVac.ExtSEIRSNetworkModelVac(G=G_Lockdown2, p=p_NationalLockdown2,
                                      beta=BETA_HeavyMaskingTeleworking,beta_asym=BETA_asym_HeavyMaskingTeleworking, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                      gamma_asym=GAMMA_asym, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H,
                                      a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,
                                      alpha=ALPHA,beta_pairwise_mode=BETA_PAIRWISE_MODE, delta_pairwise_mode=DELTA_PAIRWISE_MODE, q=0,
                                       initR=stateNumbers['recovered'],initE=stateNumbers['exposed'], initI_pre=stateNumbers['i_pre'], initI_sym=stateNumbers['i_sym'],
                                        initI_asym=stateNumbers['i_asym'], initF=stateNumbers['fatalities'],initH=stateNumbers['hospitalized'])
        checkpointsVac = {'t':[50],
                          'G': [G_HeavyMaskingTeleworking],
                          'p': [p_LeisureMass],
                          'BETA': [BETA_HeavyMaskingTeleworking]}

        model2.run(T=1,verbose=True,checkpoints=checkpointsVac)
        model2.introduce_vaccined(num_new_vaccined=90)

        model2.run(T=134)
        if(plot):model2.figure_infections(ylim=15000,plot_percentages=False)

        vactimeSeries   = model2.tseries
        vacSseries      = model2.numS
        vacEseries      = model2.numE
        vacI_preseries  = model2.numI_pre
        vacI_symseries  = model2.numI_sym
        vacI_asymseries = model2.numI_asym
        vacRseries      = model2.numR
        vacHseries      = model2.numH
        vacFseries      = model2.numF

    return(timeSeries,Sseries,Eseries,I_preseries,I_symseries,I_asymseries,Rseries,Hseries,Fseries,
            vactimeSeries,vacSseries,vacEseries,vacI_preseries,vacI_symseries,vacI_asymseries,vacRseries,vacHseries,vacFseries)
