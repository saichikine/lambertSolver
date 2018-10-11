#Imports

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt
from numpy.lib import scimath
from pylab import rcParams
rcParams['figure.figsize'] = 7.5, 7.5

import warnings; warnings.simplefilter('ignore')

#Compute Orbital Elements
def orbitalElements(R, V, mu, printResults):

    v = np.linalg.norm(V)

    #Define unit vectors
    x = [1, 0, 0]
    y = [0, 1, 0]
    z = [0, 0, 1]

    #Calculate h
    H = np.cross(R,V)
    h = np.linalg.norm(H)

    #Calculate p
    p = h**2/mu

    #Calculate e
    eVec = np.cross(V, H)/mu - (R/np.linalg.norm(R))
    e = np.linalg.norm(eVec)

    #Calculate a
    a = p/(1-e**2)

    #Calculate i
    i = np.arccos(H[2]/h)

    #Calculate RAAN (big omega)
    n = np.cross(z,H)

    if (np.dot(n,y) >= 0):
        RAAN = np.arccos(n[0]/np.linalg.norm(n))
    elif (np.dot(n,y) < 0):
        RAAN = -np.arccos(n[0]/np.linalg.norm(n))

    #Calculate argument of periapsis (little omega)
    if (np.dot(z, eVec) >= 0):
        aop = np.arccos(np.dot(n,eVec)/(np.linalg.norm(n)*e))
    elif  (np.dot(z,eVec) < 0):
        aop = 2*np.pi - np.arccos(np.dot(n,eVec)/(np.linalg.norm(n)*e))

    #Calculate true anomaly
    if (np.dot(R, V) >= 0):
        theta = np.arccos(np.dot(R,eVec)/(np.linalg.norm(R)*e))
    elif (np.dot(R, V) < 0):
        theta = 2*np.pi - np.arccos(np.dot(R,eVec)/(np.linalg.norm(R)*e))

    if printResults:
        print('Semi major axis a = ', a, ' km')
        print('Eccentricity e = ', e)
        print('Inclination angle i = ', np.degrees(i), ' degrees')
        print('Right ascension of ascending node = ', np.degrees(RAAN), ' degrees')
        print('Argument of periapsis = ', np.degrees(aop), ' degrees')
        print('True anomaly = ', np.degrees(theta), ' degrees')

    return([a, e, i, RAAN, aop, theta, h, p])

#Convert Between Theta and E
def theta2E(theta, e):
    E = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(theta/2))
    return(E)

def E2theta(E, e):
    theta = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    return(theta)

#Kepler's eqn minus dt
def Kepler(E2, info):
    #print("In Kepler: info = ", info)
    n = info[0]
    e = info[1]
    dt = info[2]*3600 #convert hours to seconds
    E1 = info[3]
    return (E2 - e*np.sin(E2) - (E1 - e*np.sin(E1)) - n*dt)

#Use f and g to Find Rf and Vf
def fgTheta(Ri, Vi, rf, mu, p, deltaTheta):

    ri = np.linalg.norm(Ri)
    vi = np.linalg.norm(Vi)

    f = 1 - ((rf/p)*(1 - np.cos(deltaTheta)))
    g = (ri*rf/np.sqrt(mu*p))*(np.sin(deltaTheta))

    fdot = np.sqrt(mu/p)*np.tan(deltaTheta/2)*(((1 - np.cos(deltaTheta))/p) - (1/rf) - (1/ri))
    gdot = 1-((ri/p)*(1-np.cos(deltaTheta)))

    R2 = np.multiply(f, Ri) + np.multiply(g, Vi)
    V2 = np.multiply(fdot, Ri) + np.multiply(gdot, Vi)

    return([R2, V2])

def fgE(Ri, Vi, rf, dt, a, mu, deltaE):

    deltat = dt*3600 #hours to seconds

    ri = np.linalg.norm(Ri)
    vi = np.linalg.norm(Vi)

    f = 1 - ((a/ri)*(1 - np.cos(deltaE)))
    g = deltat - (np.sqrt((a**3)/mu)*(deltaE - np.sin(deltaE)))

    fdot = (-np.sin(deltaE)*np.sqrt(mu*a))/(ri*rf)
    gdot = 1 - (1/rf)*(1 - np.cos(deltaE))

    R2 = np.multiply(f, Ri) + np.multiply(g, Vi)
    V2 = np.multiply(fdot, Ri) + np.multiply(gdot, Vi)

    return([R2, V2])

#Secant Method Solver
def solve(func, info, x0, x1, err, Nmax, isVerbose):

    n = 0

    while (n < Nmax):

        n = n+1
        if isVerbose:
            print("In solve: Iteration: ", n)
            print("In solve: x0 = ", x0)
            print("In solve: x1 = ", x1)

        x2 = x1 - func(x1, info)*((x1-x0)/(func(x1, info) - func(x0, info)))
        if isVerbose:
            print("In solve: x2 = ", x2)

        if (np.abs(x2 - x1) < err):
            print("\nIn solve: Successful Solve.\n")
            return x2

        else:
            x0 = x1
            x1 = x2

    return False

#Lambert's Equation Function
def TOFLambert(a, info):

    mu = info[0]
    s = info[1]
    c = info[2]
    transferLessThan180 = info[4]
    transferGreaterThan180 = not transferLessThan180
    shortWay = info[5]
    longWay = not shortWay
    isVerbose = info[6]

    n = np.sqrt(mu/(a**3))

    #Set alpha and beta

    if transferLessThan180:
        beta = 2*(np.arcsin(np.sqrt((s - c)/(2*a))))
        if isVerbose:
            print("In TOFLambert, Transfer less than 180")
    elif transferGreaterThan180:
        beta = -2*(np.arcsin(np.sqrt((s - c)/(2*a))))
        if isVerbose:
            print("In TOFLambert, Transfer greater than 180")

    if shortWay:
        alpha = 2*np.arcsin(np.sqrt(s/(2*a)))
        if isVerbose:
            print("In TOFLambert, Short way")
    elif longWay:
        alpha = 2*np.pi-2*np.arcsin(np.sqrt(s/(2*a)))
        if isVerbose:
            print("In TOFLambert, Long way")

    TOF = (1/n)*((alpha - beta) - (np.sin(alpha) - np.sin(beta)))/3600

    if isVerbose:
        print("In TOFLambert, a = ", a, ' km')
        print("In TOFLambert, alpha = ", alpha)
        print("In TOFLambert, beta = ", beta)
        print("In TOFLambert, n = ", n)
        print("In TOFLambert, s = ", s)
        print("In TOFLambert, c = ", c)
        print("In TOFLambert, TOFi = ", TOF, ' hours')

    return TOF

#Lambert's Equation Minus Desired TOF
def TOFLambertSolve(a, info):
    TOFDesired = info[3]
    return (TOFLambert(a, info) - TOFDesired)

#Lambert's Problem Solver
#state1 (list of 2 3-d vectors): initial position and velocity vectors
#state2 (list of 2 3-d vectors): final position and velocity vectors
#transferLessThan180 (boolean): is this transfer angle less than 180 degrees?
#TOF (int or float): desired time of flight (HOURS)
#info (list): relevant info, such as gravitational parameter
#isVerbose (boolean): output verbose output or not
#printResults (boolean): print results or not

def Lambert(state1, state2, transferLessThan180, TOF, info, isVerbose, printResults):

    R1 = state1[0]
    V1 = state1[1]
    R2 = state2[0]
    V2 = state2[1]

    r1 = np.linalg.norm(R1)
    v1 = np.linalg.norm(V1)
    r2 = np.linalg.norm(R2)
    v2 = np.linalg.norm(V2)

    mu = info[0]

    if transferLessThan180:
        deltaTheta = np.arccos(np.dot(R1,R2)/(r1*r2))
    else:
        deltaTheta = 2*np.pi - np.arccos(np.dot(R1,R2)/(r1*r2))

    if isVerbose:
        print("Delta theta = ", np.degrees(deltaTheta), " degrees.")


    c = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(deltaTheta))
    s = 0.5*(r1+r2+c)

    if isVerbose:
        print('c = ', c)
        print('s = ', s)

    TOFParabolic = 1/3*np.sqrt(2/mu)*(s**(3/2)-(s-c)**(3/2))/3600 #hours
    if isVerbose:
        print("Parabolic TOF = ", TOFParabolic, " hours")
        print("Your TOF = ", TOF, " hours")

    isEllipse = False
    isHyperbola = False
    isParabola = False

    if TOF > TOFParabolic:
        isEllipse = True
        if isVerbose:
            print("Elliptical Transfer.")
    elif TOF < TOFParabolic:
        isHyperbola = True
        if isVerbose:
            print("Hyperbolic Transfer.")
    else:
        isParabola = True
        if isVerbose:
            print("Parabolic Transfer.")

    #Minimum energy arc parameters
    am = s/2
    nm = np.sqrt(mu/(am**3))
    alpham = np.pi
    betam0 = 2*np.arcsin(np.sqrt((s-c)/s))

    if transferLessThan180:
        betam = betam0
    else:
        betam = -betam0

    TOFm = (1/nm)*(alpham - betam - (np.sin(alpham) - np.sin(betam)))/3600 #hours
    if isVerbose:
        print("Minimum energy ellipse am = ", am, ' km.')
        print("Minimum energy ellipse TOF = ", TOFm, ' hours.')

    #short or long way

    shortWay = False
    longWay = False

    if TOF < TOFm:
        shortWay = True
    elif TOF > TOFm:
        longWay = True
    elif TOF == TOFm:
        print("Time of flight indicates parabolic transfer.")

    if isVerbose:
        print("Short way: ", shortWay)
        print("Long way: ", longWay)

    solverVerbose = False

    infoLambert = [mu, s, c, TOF, transferLessThan180, shortWay, solverVerbose]

    #Solver params
    deltaa1 = am/100
    deltaa2 = am/75
    tolerance = 1e-7
    numberIterations = 1000

    a = solve(TOFLambertSolve, infoLambert, am+deltaa1, am+deltaa1+(np.random.rand()*deltaa1), tolerance, numberIterations, solverVerbose)
    #a = opt.fsolve(TOFLambertSolve, am+deltaa1, args=infoLambert)

    if isVerbose:
        print("In Lambert: Final a = ", a, " km")
        print("In Lambert: Final TOFi is: ", TOFLambert(a, infoLambert), ' hours')

    #Choose alpha and beta
    if transferLessThan180:
        beta = 2*(np.arcsin(np.sqrt((s - c)/(2*a))))
    else:
        beta = -2*(np.arcsin(np.sqrt((s - c)/(2*a))))

    if shortWay:
        alpha = 2*np.arcsin(np.sqrt(s/(2*a)))
    elif longWay:
        alpha = 2*np.pi-2*np.arcsin(np.sqrt(s/(2*a)))

    #Choose which ellipse
    p1 = ((4*a*(s-r1)*(s-r2))/(c**2))*(np.sin((alpha + beta)/2))**2
    p2 = ((4*a*(s-r1)*(s-r2))/(c**2))*(np.sin((alpha - beta)/2))**2
    pVec = [p1, p2]
    eVec = np.sqrt(1-(pVec/a))

    if isVerbose:
        print("Possible p values: ", pVec, " km")
        print("Possible e values: ", eVec)

    if transferLessThan180:
        if shortWay:
            eMag = min(eVec)
            p = max(pVec)
        elif longWay:
            eMag = max(eVec)
            p = min(pVec)
    else:
        if shortWay:
            eMag = max(eVec)
            p = min(pVec)
        elif longWay:
            eMag = min(eVec)
            p = max(pVec)

    if isVerbose:
        print("Selected p value = ", p, " km")
        print("Selected e value = ", eMag)

    h = np.sqrt(mu*p)
    E = mu**2*(eMag**2 - 1)/(2*h**2)

    #Find true anomalies
    theta1Vec = [np.arccos(1/eMag*(p/r1 - 1)), 2*np.pi-np.arccos(1/eMag*(p/r1 - 1))]
    theta2Vec = [np.arccos(1/eMag*(p/r2 - 1)), 2*np.pi-np.arccos(1/eMag*(p/r2 - 1))]

    combos = [theta2Vec[0] - theta1Vec[0],
             theta2Vec[1] - theta1Vec[0],
             theta2Vec[0] - theta1Vec[1],
             theta2Vec[1] - theta1Vec[1]]

    for i in range(0, len(combos)):
        if combos[i] < 0:
            combos[i] = combos[i] + 2*np.pi

    if isVerbose:
        print("Possible theta1 values: ", np.degrees(theta1Vec))
        print("Possible theta2 values: ", np.degrees(theta2Vec))
        print("Combos:")
        print(np.degrees(theta2Vec[0] - theta1Vec[0]))
        print(np.degrees(theta2Vec[1] - theta1Vec[0]))
        print(np.degrees(theta2Vec[0] - theta1Vec[1]))
        print(np.degrees(theta2Vec[1] - theta1Vec[1]))

    if(np.abs(combos[0] - deltaTheta) < 0.001):
        theta1 = theta1Vec[0]
        theta2 = theta2Vec[0]
        combosIndex = 0
    elif(np.abs(combos[1] - deltaTheta) < 0.001):
        theta1 = theta1Vec[0]
        theta2 = theta2Vec[1]
        combosIndex = 1
    elif(np.abs(combos[2] - deltaTheta) < 0.001):
        theta1 = theta1Vec[1]
        theta2 = theta2Vec[0]
        combosIndex = 2
    elif(np.abs(combos[3] - deltaTheta) < 0.001):
        theta1 = theta1Vec[1]
        theta2 = theta2Vec[1]
        combosIndex = 3

    if isVerbose:
        print("Theta1 = ", np.degrees(theta1))
        print("Theta2 = ", np.degrees(theta2))
        print("Difference = ", np.degrees(combos[combosIndex]))
        print("Delta theta = ", np.degrees(deltaTheta))

    #sigma = np.dot(R1,V1)/np.sqrt(mu)

    f = 1-((r2/p)*(1-np.cos(deltaTheta)))
    g = r1*r2/np.sqrt(mu*p)*(np.sin(deltaTheta))

    fdot = np.sqrt(mu/p)*np.tan(deltaTheta/2)*(((1-np.cos(deltaTheta))/p)-(1/r2)-(1/r1))
    #fdot = np.sqrt(mu)/(r1*p)*(sigma*(1-np.cos(deltaTheta))-np.sqrt(p)*np.sin(deltaTheta))
    gdot = 1-((r1/p)*(1-np.cos(deltaTheta)))

    V1TransInertial = (R2 - np.multiply(f, R1))/g
    V2TransInertial = np.multiply(fdot, R1) + np.multiply(gdot, V1TransInertial)

    #Delta vs
    deltav1 = V1TransInertial - V1
    deltav2 = V2 - V2TransInertial
    deltav1Mag = np.linalg.norm(deltav1)
    deltav2Mag = np.linalg.norm(deltav2)
    totalDeltav = deltav1Mag + deltav2Mag

    if printResults:
        print("In Lambert:")
        print("\nTransfer solved.\n")
        print("Initial State (inertial frame):\n")
        print("\tInitial Position = ", R1, "km")
        print("\tInitial Velocity = ", V1, "km")
        print("\tFinal Position = ", R2, "km")
        print("\tFinal Velocity = ", V2, "km\n")
        print("Transfer:\n")
        print("\tSemi-Major Axis a = ", a, " km")
        print("\tEccentricity e = ", eMag)
        print("\tOrbital Energy E = ", E, "m^2/s^2")
        print("\tTrue Anomaly at Transfer Start (theta1) = ", np.degrees(theta1), "degrees")
        print("\tTrue Anomaly at Transfer Start (theta1) = ", np.degrees(theta2), "degrees")
        print("\tInitial Transfer Arc Velocity (inertial frame) = ", V1TransInertial, "km/s")
        print("\tFinal Transfer Arc Velocity (inertial frame) = ", V2TransInertial, "km/s")
        print("\tDelta v_1 = ", deltav1Mag, "km/s")
        print("\tDelta v_2 = ", deltav2Mag, "km/s")
        print("\tTotal delta v = ", totalDeltav, "km/s")

    return([a, eMag, E, am, h, p, theta1, theta2, V1TransInertial, V2TransInertial, deltav1Mag, deltav2Mag, totalDeltav])

def CW(info, t):
    x0 = info[0]
    x0dot = info[1]
    y0 = info[2]
    y0dot = info[3]
    z0 = info[4]
    z0dot = info[5]
    n = info[6]
    x = 4*x0 + 2/n*y0dot + x0dot/n*np.sin(n*t) - (3*x0 + 2/n*y0dot)*np.cos(n*t)
    y = y0 - 2/n*x0dot - 3*(2*n*x0 + y0dot)*t + 2*(3*x0 + 2/n*y0dot)*np.sin(n*t) + 2/n*x0dot*np.cos(n*t)
    z = 1/n*z0dot*np.sin(n*t) + z0*np.cos(n*t)

    return([x, y, z])
