"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import sys
sys.path.append("../../PathPlanning/CubicSpline/")

import numpy as np
import math
import cvxpy
import matplotlib.pyplot as plt
import cubic_spline_planner

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  #state cost matrix
Qf = Q  #state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 7.2 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 2.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 0.2  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 1.5  # [m]

MAX_STEER = math.radians(45.0)  # maximum steering angle [rad]
MAX_DSTEER = math.radians(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 36 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True
show_animation_delay = True
show_graphics = False

colors=["-r","-g","-b","-e","-k"]

rtx = [0.0, 30.0/4, 30.0/4, 0.0, 0.0 ]
rty = [0.0, 0.0,  30.0/4, 30.0/4,  0.0 ]

rx = [0.0, 5.0, 10.0, 20.0]
ry = [0.0, 0.0, 0.0, 0.0]

r3x = [0.0, 6.0, 12.5, 5.0, 7.50, 3.0, -1.0]
r3y = [0.0, 0.0, 5.0, 6.50, 3.0, 5.0, -2.0]

class State:
    """
    vehicles state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None

class Robot:
    

    def __init__(self, ax, ay, color = "-k"):

        #state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        #self.angle = 0.0
        self.ax = ax
        self.ay = ay
        #self.star=star
        #self.speed_profile
        #self.speed_profiles
        #self.tr
        self.dl=1.0
        #self.goal
        #self.state
        #self.trd
        self.color=color
        
       
    def pi_2_pi(self, angle):
        while(angle > math.pi):
            angle = angle - 2.0 * math.pi

        while(angle < -math.pi):
            angle = angle + 2.0 * math.pi

        return angle
    

    def get_linear_model_matrix(self, v, phi, delta):

        A = np.matrix(np.zeros((NX, NX)))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = DT * math.cos(phi)
        A[0, 3] = - DT * v * math.sin(phi)
        A[1, 2] = DT * math.sin(phi)
        A[1, 3] = DT * v * math.cos(phi)
        A[3, 2] = DT * math.tan(delta) / WB

        B = np.matrix(np.zeros((NX, NU)))
        B[2, 0] = DT
        B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

        C = np.zeros(NX)
        C[0] = DT * v * math.sin(phi) * phi
        C[1] = - DT * v * math.cos(phi) * phi
        C[3] = v * delta / (WB * math.cos(delta) ** 2)

        return A, B, C


    def plot_car(self, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):
        if (truckcolor=="-k"):
            truckcolor=self.color
        outline = np.matrix([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -  BACKTOWHEEL, -BACKTOWHEEL], [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

        fr_wheel = np.matrix([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
     [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

	rr_wheel = np.copy(fr_wheel)

	fl_wheel = np.copy(fr_wheel)
	fl_wheel[1, :] *= -1
	rl_wheel = np.copy(rr_wheel)
	rl_wheel[1, :] *= -1

	Rot1 = np.matrix([[math.cos(yaw), math.sin(yaw)],
		              [-math.sin(yaw), math.cos(yaw)]])
	Rot2 = np.matrix([[math.cos(steer), math.sin(steer)],
		              [-math.sin(steer), math.cos(steer)]])

	fr_wheel = (fr_wheel.T * Rot2).T
	fl_wheel = (fl_wheel.T * Rot2).T
	fr_wheel[0, :] += WB
	fl_wheel[0, :] += WB

	fr_wheel = (fr_wheel.T * Rot1).T
	fl_wheel = (fl_wheel.T * Rot1).T

	outline = (outline.T * Rot1).T
	rr_wheel = (rr_wheel.T * Rot1).T
	rl_wheel = (rl_wheel.T * Rot1).T

        outline[0, :] += x
        outline[1, :] += y
        fr_wheel[0, :] += x
        fr_wheel[1, :] += y
        rr_wheel[0, :] += x
        rr_wheel[1, :] += y
        fl_wheel[0, :] += x
        fl_wheel[1, :] += y
        rl_wheel[0, :] += x
        rl_wheel[1, :] += y

        plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(x, y, "*")


    def update_state(self, state, a, delta):

        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        state.x =state.x +state.v * math.cos(state.yaw) * DT
        state.y =state.y +state.v * math.sin(state.yaw) * DT
        state.yaw =state.yaw +state.v / WB * math.tan(delta) * DT
        state.v =state.v + a * DT

        if state. v > MAX_SPEED:
           state.v = MAX_SPEED
        elif state. v < MIN_SPEED:
           state.v = MIN_SPEED
        #print delta
        return state


    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()


    def calc_nearest_index(self, state, cx, cy, cyaw, pind):

		dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
		dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

		d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

		mind = min(d)

		ind = d.index(mind) + pind

		mind = math.sqrt(mind)

		dxl = cx[ind] -state.x
		dyl = cy[ind] -state.y

		angle = self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
		if angle < 0:
		    mind *= -1

		return ind, mind


    def predict_motion(self, x0, oa, od, xref):
		xbar = xref * 0.0
		for i in range(len(x0)):
		    xbar[i, 0] = x0[i]

		state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
		for (ai, di, i) in zip(oa, od, range(1, T + 1)):
		    state = self.update_state(state, ai, di)
		    xbar[0, i] =state.x
		    xbar[1, i] =state.y
		    xbar[2, i] =state.v
		    xbar[3, i] =state.yaw

		return xbar


    def iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
		"""
		MPC contorl with updating operational point iteraitvely
		"""

		if oa is None or od is None:
		    oa = [0.0] * T
		    od = [0.0] * T

		for i in range(MAX_ITER):
		    xbar = self.predict_motion(x0, oa, od, xref)
		    poa, pod = oa[:], od[:]
		    oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
		    du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
		    if du <= DU_TH:
		        break
		else:
		    print("Iterative is max iter")

		return oa, od, ox, oy, oyaw, ov


    def linear_mpc_control(self, xref, xbar, x0, dref):
		"""
		linear mpc control

		xref: reference point
		xbar: operational point
		x0: initial state
		dref: reference steer angle
		"""

		x = cvxpy.Variable((NX, T + 1))
		u = cvxpy.Variable((NU, T))

		cost = 0.0
		constraints = []

		for t in range(T):
		    cost += cvxpy.quad_form(u[:, t], R)

		    if t != 0:
		        cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

		    A, B, C = self.get_linear_model_matrix(
		        xbar[2, t], xbar[3, t], dref[0, t])
		    constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

		    if t < (T - 1):
		        cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
		        constraints += [cvxpy.abs(u[1, t + 1] - u[1, t])
		                        <= MAX_DSTEER * DT]

		cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

		constraints += [x[:, 0] == x0]
		constraints += [x[2, :] <= MAX_SPEED]
		constraints += [x[2, :] >= MIN_SPEED]
		constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
		constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

		prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
		prob.solve(solver=cvxpy.ECOS, verbose=False)

		if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
		    ox = self.get_nparray_from_matrix(x.value[0, :])
		    oy = self.get_nparray_from_matrix(x.value[1, :])
		    ov = self.get_nparray_from_matrix(x.value[2, :])
		    oyaw = self.get_nparray_from_matrix(x.value[3, :])
		    oa = self.get_nparray_from_matrix(u.value[0, :])
		    odelta = self.get_nparray_from_matrix(u.value[1, :])

		else:
		    print("Error: Cannot solve mpc..")
		    oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

		return oa, odelta, ox, oy, oyaw, ov


    def calc_ref_trajectory(self,state, cx, cy, cyaw, ck, sp, dl, pind):
		xref = np.zeros((NX, T + 1))
		dref = np.zeros((1, T + 1))
		ncourse = len(cx)

		ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, pind)

		if pind >= ind:
		    ind = pind

		xref[0, 0] = cx[ind]
		xref[1, 0] = cy[ind]
		xref[2, 0] = sp[ind]
		xref[3, 0] = cyaw[ind]
		dref[0, 0] = 0.0  # steer operational point should be 0

		travel = 0.0

		for i in range(T + 1):
		    travel += abs(state.v) * DT
		    dind = int(round(travel / dl))

		    if (ind + dind) < ncourse:
		        xref[0, i] = cx[ind + dind]
		        xref[1, i] = cy[ind + dind]
		        xref[2, i] = sp[ind + dind]
		        xref[3, i] = cyaw[ind + dind]
		        dref[0, i] = 0.0
		    else:
		        xref[0, i] = cx[ncourse - 1]
		        xref[1, i] = cy[ncourse - 1]
		        xref[2, i] = sp[ncourse - 1]
		        xref[3, i] = cyaw[ncourse - 1]
		        dref[0, i] = 0.0

		return xref, ind, dref


    def check_goal(self,state, goal, tind, nind):

		# check goal
		dx =state.x - goal[0]
		dy =state.y - goal[1]
		d = math.sqrt(dx ** 2 + dy ** 2)

		if (d <= GOAL_DIS):
		    isgoal = True
		else:
		    isgoal = False

		if abs(tind - nind) >= 5:
		    isgoal = False

		if (abs(state.v) <= STOP_SPEED):
		    isstop = True
		else:
		    isstop = False

		if isgoal and isstop:
		    return True

		return False

    iterator_of_show_animation=0
    def do_simulation(self, cx, cy, cyaw, ck, sp, dl, initial_state):
        """
        Simulation

        cx: course x position list
        cy: course y position list
        cy: course yaw position list
        ck: course curvature list
        sp: speed profile
        dl: course tick [m]

        """
      
        goal = [cx[-1], cy[-1]]

        state = initial_state

        # initial yaw compensation
        if state.yaw - cyaw[0] >= math.pi:
             state.yaw -= math.pi * 2.0
        elif state.yaw - cyaw[0] <= -math.pi:
             state.yaw += math.pi * 2.0

        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        t = [0.0]
        d = [0.0]
        a = [0.0]
        target_ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, 0)

        odelta, oa = None, None

        cyaw = self.smooth_yaw(cyaw)

        while MAX_TIME >= time:
            xref, target_ind, dref = self.calc_ref_trajectory(
		       state, cx, cy, cyaw, ck, sp, dl, target_ind)

            x0 = [state.x,state.y,state.v,state.yaw]  # currentstate

            oa, odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(
		        xref, x0, dref, oa, odelta)

            if odelta is not None:
                di, ai = odelta[0], oa[0]

            state = self.update_state(state, ai, di)
		   
            time = time + DT

            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            v.append(state.v)
            t.append(time)
            d.append(di)
            a.append(ai)

            if self.check_goal(state, goal, target_ind, len(cx)):
                print("Goal")
                break
	
	
            if show_animation:
                plt.cla()
                if ox is not None:
                    plt.plot(ox, oy, "xr", label="MPC")
                plt.plot(cx, cy, "-r", label="course")
                plt.plot(x, y, "ob", label="trajectory")
                plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
                plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                r.plot_car(state.x,state.y,state.yaw, steer=di)
                plt.axis("equal")
                plt.grid(True)
                plt.title("Time[s]:" + str(round(time, 2)) +
		                  ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
                if(show_animation_delay):
                    plt.pause(0.00001)

        return t, x, y, yaw, v, d, a


    def calc_speed_profile(self, cx, cy, cyaw, target_speed):

		speed_profile = [target_speed] * len(cx)
		direction = 1.0  # forward

		# Set stop point
		for i in range(len(cx) - 1):
		    dx = cx[i + 1] - cx[i]
		    dy = cy[i + 1] - cy[i]

		    move_direction = math.atan2(dy, dx)

		    if dx != 0.0 and dy != 0.0:
		        dangle = abs(self.pi_2_pi(move_direction - cyaw[i]))
		        if dangle >= math.pi / 4.0:
		            direction = -1.0
		        else:
		            direction = 1.0

		    if direction != 1.0:
		        speed_profile[i] = - target_speed
		    else:
		        speed_profile[i] = target_speed

		speed_profile[-1] = 0.0

		return speed_profile


    def smooth_yaw(self, yaw):

		for i in range(len(yaw) - 1):
		    dyaw = yaw[i + 1] - yaw[i]

		    while dyaw >= math.pi / 2.0:
		        yaw[i + 1] -= math.pi * 2.0
		        dyaw = yaw[i + 1] - yaw[i]

		    while dyaw <= -math.pi / 2.0:
		        yaw[i + 1] += math.pi * 2.0
		        dyaw = yaw[i + 1] - yaw[i]

		return yaw

    def get_straight_coursek(self, dl):

        ax = self.ax
        ay = self.ay
        #ax = [0.0, 30.0/4, 30.0/4, 0.0, 0.0 ]
        #ay = [0.0, 0.0,  30.0/4, 30.0/4,  0.0 ]
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

        return cx, cy, cyaw, ck


    def get_straight_cour (self, dl):
        ax = self.ax   #[0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        ay = self.ay   #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
		    ax, ay, ds=dl)

        return cx, cy, cyaw, ck

#pair <pair <int,int>, pair <int, int>>
#class traektory_settings():    
class Tr():
    def __init__(self, cx, cy, cyaw, ck):
        self.cx, self.cy, self.cyaw, self.ck = cx, cy, cyaw, ck
    
class Trd(): #Data of traektory
    def __init__(self, state):
        self.time = 0.0
        self.x = [state.x]
        self.y = [state.y]
        self.yaw = [state.yaw]
        self.v = [state.v]
        self.t = [0.0]
        self.d = [0.0]
        self.a = [0.0]
        #self.target_ind
        self.odelta, self.oa = None, None
        #self.xref
        #self.dref
        self.goall=0
        #self.x0
        #self.oa, self.odelta, self.ox, self.oy, self.oyaw, self.ov
        #self.di, self.ai
        
        
        
        
        
        
    def __init__(self):
        #self.time
        #self.x
        #self.y
        #self.yaw
        #self.v
        #self.t
        #self.d
        #self.a
        #self.target_ind
        self.odelta, self.oa = None, None
        #self.xref
        #self.dref
        self.goall=0
        #self.x0
        #self.oa, self.odelta, self.ox, self.oy, self.oyaw, self.ov
        #self.di, self.ai
        
        
class Robot_group(Robot):
    

    def __init__(self, group = []):

        #state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        #self.angle = 0.0
        self.group=group #massiv of robots
        #self.star=star
        #{1,2,3,4,"afsdafs",23}
        
        for i in self.group:
            i.tr=(Tr(i.get_straight_coursek(i.dl)[0],i.get_straight_coursek(i.dl)[1],i.get_straight_coursek(i.dl)[2],i.get_straight_coursek(i.dl)[3])) #cx, cy, cyaw, ck
        #for j in range (0,10):
        #   group[j].tr=
        
        for i in range(0, len(self.group)):
            self.group[i].goal=([self.group[i].tr.cx[-1], self.group[i].tr.cy[-1]])
            self.group[i].state=(State(x=self.group[i].tr.cx[0], y=self.group[i].tr.cy[0], yaw=self.group[i].tr.cyaw[0], v=0.0))
                # initial yaw compensation
            if self.group[i].state.yaw - self.group[i].tr.cyaw[0] >= math.pi:
                 self.group[i].state.yaw -= math.pi * 2.0
            elif self.group[i].state.yaw - self.group[i].tr.cyaw[0] <= -math.pi:
                 self.group[i].state.yaw += math.pi * 2.0

       
    def do_simulation(self):
        """
        Simulation

        cx: course x position list
        cy: course y position list
        cy: course yaw position list
        ck: course curvature list
        sp: speed profile
        dl: course tick [m]

        """
        
        for r in self.group:
            r.trd=Trd()

            r.trd.time = 0.0
            r.trd.x = [r.state.x]
            r.trd.y = [r.state.y]
            r.trd.yaw = [r.state.yaw]
            r.trd.v = [r.state.v]
            r.trd.t = [0.0]
            r.trd.d = [0.0]
            r.trd.a = [0.0]
            r.trd.target_ind, _ = r.calc_nearest_index(r.state, r.tr.cx, r.tr.cy, r.tr.cyaw, 0)

            r.trd.odelta, r.trd.oa = None, None

            r.tr.cyaw = r.smooth_yaw(r.tr.cyaw)
            r.trd.goall=False
        
        while MAX_TIME >= r.trd.time:
            flag=True
            count=0
            for r in self.group:
                r.trd.xref, r.trd.target_ind, r.trd.dref = r.calc_ref_trajectory(
                   r.state, r.tr.cx, r.tr.cy, r.tr.cyaw, r.tr.ck, r.speed_profile, r.dl, r.trd.target_ind)

                r.trd.x0 = [r.state.x, r.state.y, r.state.v, r.state.yaw]  # currentstate

                r.trd.oa, r.trd.odelta, r.trd.ox, r.trd.oy, r.trd.oyaw, r.trd.ov = self.iterative_linear_mpc_control(
                    r.trd.xref, r.trd.x0, r.trd.dref, r.trd.oa, r.trd.odelta)

                if r.trd.odelta is not None:
                    r.trd.di, r.trd.ai = r.trd.odelta[0], r.trd.oa[0]

                r.state = self.update_state(r.state, r.trd.ai, r.trd.di)
	           
                r.trd.time = r.trd.time + DT

                r.trd.x.append(r.state.x)
                r.trd.y.append(r.state.y)
                r.trd.yaw.append(r.state.yaw)
                r.trd.v.append(r.state.v)
                r.trd.t.append(r.trd.time)
                r.trd.d.append(r.trd.di)
                r.trd.a.append(r.trd.ai)
                
                if (r.trd.goall):
                    continue
                if r.check_goal(r.state, r.goal, r.trd.target_ind, len(r.tr.cx)):
                    r.trd.goall=True
                    continue
                flag=False
                count+=1
                
            print(count)
            if(flag):
                print("Goal")
                break
               
            if show_animation:
                plt.cla()
                for r in self.group:
                    if r.trd.ox is not None:
                        plt.plot(r.trd.ox, r.trd.oy, "xr", label="MPC")
                    plt.plot(r.tr.cx, r.tr.cy, "-r", label="course")
                    plt.plot(r.trd.x, r.trd.y, "ob", label="trajectory")
                    plt.plot(r.trd.xref[0, :], r.trd.xref[1, :], "xk", label="xref")
                    plt.plot(r.tr.cx[r.trd.target_ind], r.tr.cy[r.trd.target_ind], "xg", label="target")
                    r.plot_car(r.state.x,r.state.y,r.state.yaw, steer=r.trd.di)
                    plt.axis("equal")
                    plt.grid(True)
                    plt.title("Time[s]:" + str(round(r.trd.time, 2)) +
	                          ", speed[km/h]:" + str(round(r.state.v * 3.6, 2)))
                if(show_animation_delay):
                    plt.pause(0.00001)
            
            
        #return t, x, y, yaw, v, d, a

    #Mozzhno dobavit v class Robot
    def calc_speed_profile(self, target_speed):
        #self.speed_profiles=[]
        for r in self.group:
            speed_profile = [target_speed] * len(r.tr.cx)
    	    direction = 1.0  # forward
            #cx, cy, cyaw, nichego_ne_nuzhno = r.get_straight_coursek(dl)
		    # Set stop point
            for i in range(len(r.tr.cx) - 1):
                dx = r.tr.cx[i + 1] - r.tr.cx[i]
                dy = r.tr.cy[i + 1] - r.tr.cy[i]

                move_direction = math.atan2(dy, dx)

                if dx != 0.0 and dy != 0.0:
                    dangle = abs(self.pi_2_pi(move_direction - r.tr.cyaw[i]))
                    if dangle >= math.pi / 4.0:
                        direction = -1.0
                    else:
                        direction = 1.0

                if direction != 1.0:
                    speed_profile[i] = - target_speed
                else:
                    speed_profile[i] = target_speed

		    speed_profile[-1] = 0.0
            #speed_profiles.append(speed_profile)
            r.speed_profile=speed_profile
		#return speed_profiles


def main():
    print(__file__ + " start!!")
    robot = Robot(rtx,rty)
    rob = Robot(rx,ry)
    dl = 1.0  # course tick
    rob_group = [robot,rob]
    rob_group_out = []
    
    for iter in range(0,len(rob_group)):
        #cx, cy, cyaw, ck = get_straight_course(dl)
        cx, cy, cyaw, ck = rob_group[iter].get_straight_coursek(dl)
        #cx, cy, cyaw, ck = rob.get_straight_cour(dl)
		#cx, cy, cyaw, ck = get_straight_course2(dl)
		#cx, cy, cyaw, ck = get_straight_course3(dl)
		#cx, cy, cyaw, ck = get_forward_course(dl)
		#cx, cy, cyaw, ck = get_switch_back_course(dl)

        sp = rob_group[iter].calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
        #sp = rob.calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

        initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

        #t, x, y, yaw, v, d, a = robot.do_simulation(
		    #cx, cy, cyaw, ck, sp, dl, initial_state)

        rob_group_out.append(rob_group[iter].do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state))
		
	   
		    
        """
        delta_d = np.array(d).round(2)
        print delta_d
        print len(delta_d)
        out_str="int traektor_size="+str(len(delta_d))+";\nfloat traektor[]={ "
        for i in delta_d:
            out_str+=str(i)+", "
            out_str=out_str[0:-2]+"};"
        print(out_str)
        f=open("out_line_trajektory.txt", "w")
        f.write(out_str)
        f.close()
        """
        print("Ja zakonchil zapis")
        
        if show_graphics:
            plt.close("all")
            plt.subplots()
            plt.plot(cx, cy, "-r", label="spline")
            plt.plot(x, y, "-g", label="tracking")
            plt.grid(True)
            plt.axis("equal")
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.legend()

            plt.subplots()
            plt.plot(t, v, "-r", label="speed")
            plt.grid(True)
            plt.xlabel("Time [s]")
            plt.ylabel("Speed [kmh]")

            """
		    plt.subplots()
		    plt.plot(t, delta_d, "-r", label="angle")
		    plt.grid(True)
		    plt.xlabel("Time [s]")
		    plt.ylabel("Rad")
            """
            plt.show()
	   
		
        
def main2():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_straight_course(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=0.0, v=0.0)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    if show_animation:
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


def main3():
    print(__file__ + " start!!")
    robot = Robot(rtx,rty,"-r")
    rob = Robot(rx,ry,"-b")
    robot3 = Robot(r3x,r3y,"-g")
    dl = 1.0  # course tick
    #rob_group = [robot,rob]
    rob_group=Robot_group([robot,rob,robot3])
    rob_group_out = []
    
    #for iter in range(0,len(rob_group.group)):
    for i in range(0,1):
        #cx, cy, cyaw, ck = get_straight_course(dl)
        #cx, cy, cyaw, ck = rob_group.get_straight_coursek(dl)
        #cx, cy, cyaw, ck = rob.get_straight_cour(dl)
		#cx, cy, cyaw, ck = get_straight_course2(dl)
		#cx, cy, cyaw, ck = get_straight_course3(dl)
		#cx, cy, cyaw, ck = get_forward_course(dl)
		#cx, cy, cyaw, ck = get_switch_back_course(dl)

        #sp = rob_group.calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
        #sp = rob.calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
        rob_group.calc_speed_profile(TARGET_SPEED)
        #initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

        #t, x, y, yaw, v, d, a = robot.do_simulation(
		    #cx, cy, cyaw, ck, sp, dl, initial_state)

        #rob_group_out.append(rob_group.do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state))
        rob_group_out.append(rob_group.do_simulation())
		
	   
		    
        """
        delta_d = np.array(d).round(2)
        print delta_d
        print len(delta_d)
        out_str="int traektor_size="+str(len(delta_d))+";\nfloat traektor[]={ "
        for i in delta_d:
            out_str+=str(i)+", "
            out_str=out_str[0:-2]+"};"
        print(out_str)
        f=open("out_line_trajektory.txt", "w")
        f.write(out_str)
        f.close()
        """
        print("Ja zakonchil zapis")
        
        if show_graphics:
            plt.close("all")
            plt.subplots()
            plt.plot(cx, cy, "-r", label="spline")
            plt.plot(x, y, "-g", label="tracking")
            plt.grid(True)
            plt.axis("equal")
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.legend()

            plt.subplots()
            plt.plot(t, v, "-r", label="speed")
            plt.grid(True)
            plt.xlabel("Time [s]")
            plt.ylabel("Speed [kmh]")

            """
		    plt.subplots()
		    plt.plot(t, delta_d, "-r", label="angle")
		    plt.grid(True)
		    plt.xlabel("Time [s]")
		    plt.ylabel("Rad")
            """
            plt.show()

if __name__ == '__main__':
    print("Ja zakonchil zapis")
    main3()
    #main2()
