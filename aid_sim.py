"""
aid_sim.py - Controllori AID e simulazione in silico con simglucose.

Scelte fondamentali (da dichiarare nella tesi):
  * Passo di campionamento: dt = env.sample_time = 3 min (sensore Dexcom), 320 passi = 16 ore.
  * Il pasto e' un disturbo NON annunciato: nel CustomScenario un tempo float e' un OFFSET in ore
    dall'istante di inizio. Con inizio alle 08:00, offset 4.0 h = pasto alle ore 12:00.
  * Le metriche sono calcolate sulla glicemia PLASMATICA (info['bg']), il controllore vede solo il CGM.
  * IOB stimato = S1 + S2 (insulina sottocutanea non ancora assorbita), in unita' [U].
  * Ipoglicemia tardiva = TBR nella finestra che parte 3 ore dopo il pasto e arriva a fine simulazione.
"""
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from simglucose.simulation.env import T1DSimEnv
from simglucose.actuator.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Controller, Action

START = datetime(2026, 1, 1, 8, 0, 0)
MEAL_OFFSET_H = 4.0            # pasto 4 h dopo l'inizio = ore 12:00

class AIDController(Controller):
    """PID a tempo discreto (U/min). Moduli opzionali: tetto IOB e sospensione predittiva (PLGS)."""
    def __init__(self, basal_rate, dt, target=120.0, kp=1e-4, ki=5e-6, kd=2e-3,
                 i_min=-2000.0, i_max=10000.0, use_iob=False, use_plgs=False,
                 iob_alpha=1.5, tau_s=50.0, t_pred=20.0, plgs_pred=90.0, plgs_now=75.0):
        self.ub=basal_rate; self.dt=dt; self.target=target
        self.kp=kp; self.ki=ki; self.kd=kd; self.i_min=i_min; self.i_max=i_max
        self.use_iob=use_iob; self.use_plgs=use_plgs
        self.iob_max=iob_alpha*basal_rate*60.0     # U
        self.tau_s=tau_s; self.t_pred=t_pred; self.plgs_pred=plgs_pred; self.plgs_now=plgs_now
        self.reset()
    def reset(self):
        self.I=0.0; self.prev=None; self.s1=0.0; self.s2=0.0; self.iob=0.0
    def policy(self, observation, reward, done, **info):
        y=observation.CGM; e=y-self.target
        slope=0.0 if self.prev is None else (y-self.prev)/self.dt      # mg/dL/min
        self.prev=y
        self.I=float(np.clip(self.I+e*self.dt,self.i_min,self.i_max))
        u=self.ub+self.kp*e+self.ki*self.I+self.kd*slope
        self.iob=self.s1+self.s2                                       # U (insulina sottocutanea non ancora assorbita)
        if self.use_iob and self.iob>self.iob_max and u>self.ub: u=self.ub
        if self.use_plgs and (y+slope*self.t_pred<self.plgs_pred or y<self.plgs_now): u=0.0
        u=max(0.0,min(u,0.25))
        ex=max(0.0,u-self.ub)
        d1=(ex-self.s1/self.tau_s)*self.dt; d2=(self.s1/self.tau_s-self.s2/self.tau_s)*self.dt
        self.s1+=d1; self.s2+=d2
        return Action(basal=u,bolus=0.0)

def simulate(patient_name, cho, ctrl_kwargs, hours=16.0, seed=1):
    p=T1DPatient.withName(patient_name); s=CGMSensor.withName('Dexcom',seed=seed); pu=InsulinPump.withName('Insulet')
    sc=CustomScenario(start_time=START,scenario=[(MEAL_OFFSET_H,cho)] if cho>0 else [])
    env=T1DSimEnv(p,s,pu,sc); dt=env.sample_time
    ub=p._params['u2ss']*p._params['BW']/6000.0
    c=AIDController(ub,dt,**ctrl_kwargs)
    obs,r,d,info=env.reset(); c.reset(); rows=[]
    for k in range(int(hours*60/dt)):
        a=c.policy(obs,r,d,**info)
        rows.append((k*dt/60.0,obs.CGM,None,a.basal*60.0,c.iob))
        obs,r,d,info=env.step(a)
        rows[-1]=(rows[-1][0],rows[-1][1],info['bg'],rows[-1][3],rows[-1][4])
    return pd.DataFrame(rows,columns=['t_h','CGM','BG','Insulin_Uh','IOB']),ub

def metrics(df, signal='BG', meal_h=MEAL_OFFSET_H):
    g=df[signal].values; n=len(g)
    late=df[df.t_h>=meal_h+3][signal].values
    return {'TIR':100*np.mean((g>=70)&(g<=180)),'TAR':100*np.mean(g>180),'TAR>250':100*np.mean(g>250),
            'TBR':100*np.mean(g<70),'TBR<54':100*np.mean(g<54),'TBR_tardivo':100*np.mean(late<70),
            'picco':g.max(),'nadir':g.min()}
