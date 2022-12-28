import taichi as ti
import taichi.math as tm

ti.init(arch=ti.metal,debug=True)

#cloth with mass-spring system
#particle
n = 11
mass = 1
X = ti.Vector.field(2, float, shape=(n, n))
V = ti.Vector.field(2, float, shape=(n, n))
F = ti.Vector.field(2, float, shape=(n, n))

#environment
dt = 0.033
damp = 0.99
quad_size = 1 / n

#controler
serial = True
paused = ti.field(int, shape=())

#triangle
#the number of triangles
t = (n - 1) * (n - 1) * 2
#the indices of every mass in every triangle
T = ti.field(int, shape=t*3)

#spring
stiffness = 3e4
#every edge need to be represented by two vertices
Temp_Edge = ti.field(int, shape=t*3*2)
e_number = ti.field(int, shape=())
Edge = ti.field(int, shape=t*3*2)
#every edge with a value of length
Length = ti.field(float, shape=t*3)

#for gui
Vertices1 = ti.Vector.field(2, float, t)
Vertices2 = ti.Vector.field(2, float, t)
Vertices3 = ti.Vector.field(2, float, t)
color_t = ti.Vector.field(3, float, t)

#for PBD
X_new = ti.Vector.field(2, float, shape=(n, n))
num = ti.field(int, shape=(n, n))

@ti.func
def Swap(a, b):
    return b, a

@ti.func
def to_Vertices(X):
    for i, j in ti.ndrange(n-1, n-1):
        quad_id = i * (n - 1) + j
        #1st triangle
        Vertices1[quad_id * 2 + 0] = X[i, j]
        Vertices2[quad_id * 2 + 0] = X[i, j+1]
        Vertices3[quad_id * 2 + 0] = X[i+1, j]
        #2nd triangle
        Vertices1[quad_id * 2 + 1] = X[i+1, j+1]
        Vertices2[quad_id * 2 + 1] = X[i, j+1]
        Vertices3[quad_id * 2 + 1] = X[i+1, j]


@ti.kernel
def Paint_Triangles():
    for i in range((n-1)*(n-1)):

        color_t[2*i+0] = [255, 182, 193]
        color_t[2*i+1] = [0, 191, 255]


@ti.kernel
def Start():
    for i, j in X:
        X[i, j] = [(i * quad_size - 0.5) * 0.5, (j * quad_size + 0.5) ]
        X[i, j] += [0.6, 0]

    for i, j in ti.ndrange(n-1, n-1):
        quad_id = i * (n - 1) + j
        #1st triangle in the current quad
        T[quad_id*6+0] = i * n + j
        T[quad_id*6+1] = i * n + (j + 1)
        T[quad_id*6+2] = (i + 1) * n + j
        #2nd triangle in the current quad
        T[quad_id*6+3] = (i + 1) * n + (j + 1)
        T[quad_id*6+4] = i * n + (j + 1)
        T[quad_id*6+5] = (i + 1) * n + j
        #print([T[quad_id*6+0],T[quad_id*6+1],T[quad_id*6+2],T[quad_id*6+3],T[quad_id*6+4],T[quad_id*6+5]])
        to_Vertices(X)

    #for k in range(1):
    for i in range(0, t):
        Temp_Edge[i*3*2+0] = T[i*3+0]
        Temp_Edge[i*3*2+1] = T[i*3+1]
        Temp_Edge[i*3*2+2] = T[i*3+2]
        Temp_Edge[i*3*2+3] = T[i*3+1]
        Temp_Edge[i*3*2+4] = T[i*3+2]
        Temp_Edge[i*3*2+5] = T[i*3+0]
        #print([Temp_Edge[i*3*2+0],Temp_Edge[i*3*2+1],Temp_Edge[i*3*2+2], Temp_Edge[i*3*2+3],Temp_Edge[i*3*2+4],Temp_Edge[i*3*2+5]])

    #if serial:
    for i in range(0, t*3):
        if Temp_Edge[2*i+0] > Temp_Edge[2*i+1]:
            Temp_Edge[2*i+0], Temp_Edge[2*i+1] = Swap(Temp_Edge[2*i+0], Temp_Edge[2*i+1])
            #print([Temp_Edge[2*i+0], Temp_Edge[2*i+1]])
    #reorder the Temp_Edge by a simple serial bubble sort
    for k in range(1):
        for i in range(2, t*3):
            for j in range(0, t*3-i*2):
                if Temp_Edge[2*j+0] > Temp_Edge[2*j+2]:
                    Temp_Edge[2*j+0], Temp_Edge[2*j+2] = Swap(Temp_Edge[2*j+0], Temp_Edge[2*j+2])
                    Temp_Edge[2*j+1],Temp_Edge[2*j+3] = Swap(Temp_Edge[2*j+1], Temp_Edge[2*j+3])
                elif Temp_Edge[2*j+0] == Temp_Edge[2*j+2]:
                    if Temp_Edge[2*j+1] > Temp_Edge[2*j+3]:
                        Temp_Edge[2 * j + 0], Temp_Edge[2 * j + 2] = Swap(Temp_Edge[2 * j + 0], Temp_Edge[2 * j + 2])
                        Temp_Edge[2 * j + 1], Temp_Edge[2 * j + 3] = Swap(Temp_Edge[2 * j + 1], Temp_Edge[2 * j + 3])

    e_number[None] = 0
    for i in range(0, t*3):
        if (2*i)==0 or Temp_Edge[2*i+0] != Temp_Edge[2*i-2] or Temp_Edge[2*i+1] != Temp_Edge[2*i-1]:
            e_number[None] += 1

    #e = 0
    for k in range(1):
        e = 0
        for i in range(0, t*3):
            if (2*i)==0 or Temp_Edge[2*i+0] != Temp_Edge[2*i-2] or Temp_Edge[2*i+1] != Temp_Edge[2*i-1]:
                Edge[e*2+0] = Temp_Edge[2*i+0]
                Edge[e*2+1] = Temp_Edge[2*i+1]
                e += 1

    for e in range(0, e_number[None]/2):
        vi = Edge[2*e+0] // n
        vj = Edge[2*e+0] % n
        ui = Edge[2*e+1] // n
        uj = Edge[2*e+1] % n
        Length[e] = tm.dot(X[vi, vj] - X[ui, uj], X[vi, vj] - X[ui, uj])
        Length[e] = tm.sqrt(Length[e])
        #print(Length[2*e])

    for i, j in ti.ndrange(n, n):
        V[i, j] = [0, 0]
        F[i, j] = [0, 0]


#Sphere
Sphere_center = ti.Vector.field(2, dtype=float, shape=())
Sphere_radius = 0.3
Sphere_center[None] = [0.3, 0]
'''
center = [0.25, 0.5]
radius = 0.1
'''

#solve PBD with Jacobi
@ti.kernel
def PBD():
    alpha = 0.2
    #X_new = ti.Vector.field(2, float, shape=(n, n))
    #num = ti.field(int, shape=(n, n))
    for i, j in ti.ndrange(n, n):
        X_new[i, j] = [0, 0]
        num[i, j] = 0

    for e in range(0, e_number[None]/2):
        vi = Edge[e*2+0] // n
        vj = Edge[e*2+0] % n
        ui = Edge[e*2+1] // n
        uj = Edge[e*2+1] % n
        currentLength = tm.sqrt(tm.dot(X[vi, vj] - X[ui, uj], X[vi, vj] - X[ui, uj]))
        X_new[vi, vj] += X[vi, vj] - 0.5 * (currentLength - Length[e]) * (X[ui, uj] - X[ui, uj]) / currentLength
        X_new[ui, uj] += X[ui, uj] + 0.5 * (currentLength - Length[e]) * (X[ui, uj] - X[ui, uj]) / currentLength
        num[vi, vj] += 1
        num[ui, uj] += 1
        #print(currentLength)

    for i, j in ti.ndrange(n, n):
        temp = X[i, j]
        X[i, j] = (X_new[i, j] + alpha * X[i, j]) / (num[i, j] + alpha)
        X_new[i, j] = X[i, j] - temp

    for i, j in ti.ndrange(n, n):
        if i == (n - 1) and j == 0:
            V[i, j] = [0, 0]
        else:
            V[i, j] = X_new[i, j] / dt
        #print(X[i, j])

    to_Vertices(X)


@ti.kernel
def GS():
    for i, j in ti.ndrange(n, n):
        X_new[i, j] = [0, 0] #Here, X_new actually represents the delta X

    for e in range(0, e_number[None]/2):
        vi = Edge[e*2+0] // n
        vj = Edge[e*2+0] % n
        ui = Edge[e*2+1] // n
        uj = Edge[e*2+1] % n
        currentLength = tm.sqrt(tm.dot(X[vi, vj] - X[ui, uj], X[vi, vj] - X[ui, uj]))
        X_new[vi, vj] = - 0.5 * (currentLength - Length[e]) * (X[ui, uj] - X[ui, uj]) / currentLength
        X_new[ui, uj] = + 0.5 * (currentLength - Length[e]) * (X[ui, uj] - X[ui, uj]) / currentLength

        ti.atomic_add(X[vi, vj], X_new[vi, vj])
        ti.atomic_add(X[ui, uj], X_new[ui, uj])

    for i, j in ti.ndrange(n, n):
        if i == (n - 1) and j == 0:
            V[i, j] = [0, 0]
        else:
            V[i, j] = X_new[i, j] / dt

    to_Vertices(X)


@ti.kernel
def Collision_Handle():
    for i, j in ti.ndrange(n, n):

        if i == (n-1) and j == 0:
            V[i, j] = [0, 0]

        else:
            res = X[i, j] - Sphere_center[None]
            norm = tm.sqrt(tm.dot(res, res))

            #collision with spere
            if norm < Sphere_radius:
                V[i, j] += (Sphere_center[None] + Sphere_radius * res / norm - X[i, j])
                X[i, j] = Sphere_center[None] + Sphere_radius * res / norm

            #collision with floor

            if X[i, j].y < 0:
                V[i, j].y += (0 - X[i, j].y) / dt
                #V[i, j].y += (0 - X[i, j].y) / dt , which will be bounced by the floor
                X[i, j].y = 0


    to_Vertices(X)


@ti.kernel
def Update():
    for i, j in ti.ndrange(n, n):
        F[i, j] = [0, -9.8 * mass]
        #print(F[i, j])

    for i, j in ti.ndrange(n, n):

        if i == (n-1) and j == 0:
            V[i, j] = [0, 0]

        else:
            V[i, j] *= damp
            V[i, j] += dt * F[i, j] / mass
            #print(V[i, j])

        X[i, j] += V[i, j] * dt
        #print(X[i, j])

    to_Vertices(X)


def main():
    gui = ti.GUI("2D PBD for cloth", res=(500, 500))

    gui.x = -0.5
    Paint_Triangles()
    Start()
    while gui.running:

        for e in gui.get_events(ti.GUI.PRESS):

            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                gui.running = False
            if e.key == 'r':
                Start()
            elif e.key == ti.GUI.SPACE:
                paused[None] = not paused[None]

        if not paused[None]:
            Update()

            for k in range(32):
                #PBD()
                GS()

            Collision_Handle()

        gui.circle(pos=Sphere_center[None], color=0x00BFFF ,radius=Sphere_radius)
        gui.circles(pos=Vertices1.to_numpy(), color=0x1E90FF, radius=3)
        gui.circles(pos=Vertices2.to_numpy(), color=0x1E90FF, radius=3)
        gui.circles(pos=Vertices3.to_numpy(), color=0x1E90FF, radius=3)
        gui.triangles(a=Vertices1.to_numpy(), b=Vertices2.to_numpy(), c=Vertices3.to_numpy(), color=0xFFB6C1)

        gui.show()


if __name__ == '__main__':
    main()

