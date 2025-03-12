import numpy as np
import random,math,pygame,time

#short afternoon project implementing boids 
#not the cleanest but it works

SIZE = (800,800)
BOUNDS = (-50.0,-50.0,850.0,850.0)
boids = []

DEBUG_DRAW = False
MAX_SUBDIVISIONS = 4
NUM_BOIDS = 500
MIN_BOID_SIZE = 3
MAX_BOID_SIZE = 6

FLOCKING_WEIGHT = 0.3
ALIGNMENT_WEIGHT = 0.4
SEPARATION_WEIGHT = 0.3
SEPARATION_RANGE = 30


screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Py Boids")
run = True

deltaTime = 0
last = time.time()
quadtree = []


def mag(v):
    return np.linalg.norm(v)
def norm(v):
    m = mag(v)
    if (abs(m) < 0.00001):
        return np.array((0,0))
    return v/m
    

def pingPong(x,lower,upper):
    if (x>upper):
        x = lower+(x-upper)%(upper-lower)
    if (x<lower):
        x = upper+(x-lower)%(upper-lower)
    return x

def gradientSample(t):
    t *= math.pi

    r = 255 * abs(math.sin(t + math.pi * 0.5/3.0))
    g = 255 * abs(math.sin(t + math.pi * 1.5/3.0))
    b = 255 * abs(math.sin(t + math.pi * 2.5/3.0))
    
    return (r,g,b)
    


def createBoid(x=400,y=400,dx=1.0,dy=0.0):

    boids.append([
        #position
        np.array((float(x),float(y))),
        #direction
        np.array((float(dx),float(dy))),
        #velocity
        np.array((0.0,0.0)),
        #size
        random.randint(MIN_BOID_SIZE,MAX_BOID_SIZE),
        #random seed
        random.randint(0,1000),
        #swim pulse timer
        random.randint(0,1000)/1000.0
        
        
    ])

def updateBoid(index,dt,neighbors):

    #kinematics
    
    boids[index][0] += boids[index][2] * dt

    boids[index][0][0] = pingPong(boids[index][0][0],-10,810)
    boids[index][0][1] = pingPong(boids[index][0][1],-10,810)

    speed = mag(boids[index][2])
    
    speed = max(speed - ((speed**2) * 0.01 * dt),0)
    
    boids[index][2] = norm(boids[index][1]) * speed

    if (boids[index][5] <= 0):
        boids[index][2] += (boids[index][1] * 300.0/np.linalg.norm(boids[index][1]))*dt
        if (boids[index][5] <= -0.05):
            boids[index][5] = 1.0        
        
    boids[index][5] -= dt

    #steering

    separation = np.array((0.0,0.0))
    flocking = np.array((0.0,0.0))
    alignment = np.array((0.0,0.0))
    

    for n in neighbors:
        flocking += boids[n][0]
        if (n != index):
            deviation = (boids[n][0] - boids[index][0])
            distance = mag(deviation)
 
            alignment += boids[n][1]
            avoidance = -10000*abs(min(distance-SEPARATION_RANGE,0)/SEPARATION_RANGE)
            separation += deviation*avoidance

    separation = norm(separation) * SEPARATION_WEIGHT
    alignment =  norm(alignment/(len(neighbors))) * ALIGNMENT_WEIGHT
    flocking = norm( (flocking/(len(neighbors))) - boids[index][0] ) * FLOCKING_WEIGHT

    boids[index][1] = norm(boids[index][1] + (alignment + separation + flocking)*dt*10)
            

    

def drawBoid(boid):

    angle = math.atan2(boid[1][1],boid[1][0])
 
    offset = math.pi * (0.65 + (math.sin(boid[5]*math.pi)**2*0.3) )

    
    a = boid[0]+(np.array((math.cos(angle),math.sin(angle)))*boid[3])
    b = boid[0]+(np.array((math.cos(angle+offset),math.sin(angle+offset)))*boid[3])
    c = boid[0]
    d = boid[0]+(np.array((math.cos(angle-offset),math.sin(angle-offset)))*boid[3])
    
    pygame.draw.polygon(screen,gradientSample(boid[4]/1000),(a,b,c,d))
    pygame.draw.polygon(screen,(0,0,0),(a,b,c,d),1)


for i in range(NUM_BOIDS):
    angle = random.randint(0,1000)/1000.0 * math.pi*2
    createBoid(random.randint(int(BOUNDS[0]),int(BOUNDS[2])),random.randint(int(BOUNDS[1]),int(BOUNDS[3])),math.cos(angle),math.sin(angle))



## spatially partition boids for efficient nn search
## split space into 4 quadrants
##
##          UL | UR
##          -------
##          LL | LR
##
## each node is structured like: [coordinate,elem_1,elem_2... elem_n]



def addQuadtreeElement(element,position):
    global quadtree
    
    bounds = np.array((BOUNDS[2]-BOUNDS[0],BOUNDS[3]-BOUNDS[1]))
    center = np.array((BOUNDS[0],BOUNDS[1]))+(bounds/2.0)

    current = quadtree

    location = (0.0,0.0)
    
    for i in range(MAX_SUBDIVISIONS):
        
        index = 1
        bounds /= 2.0
        
        if (position[1] >= center[1]):
            index += 2
            center[1] += bounds[1]/2.0
        else:
            center[1] -= bounds[1]/2.0
            
        if (position[0] >= center[0]):
            center[0] += bounds[0]/2.0
        else:
            index += 1
            center[0] -= bounds[0]/2.0
        
        ## old method (very memory inefficient, created a new list at each node in the tree)
        ## doing it like that was potentially more useful in general but for nn search it was bad

        #are we at a leaf?
        #if (i==(MAX_SUBDIVISIONS-1)):
        #    if (current[index] == None):
        #        current[index] = [True,[element]]
        #        location = center
        #    else:
        #        pass
        #        #current[index][1].append(element)
        #else:
        #     if (current[index] == None):
        #        current[index] = [False,None,None,None,None]
        #current = current[index]

        location = center

    for i in range(len(quadtree)):
        if (quadtree[i][0][0] == center[0] and quadtree[i][0][1] == center[1]):
            quadtree[i].append(element)
            return

    quadtree.append([center,element])

    
def generateQuadtree():
    global quadtree
    
    #quadtree = [False,None,None,None,None]
    quadtree = []
    for i in range(len(boids)):
        addQuadtreeElement(i,boids[i][0])

def drawQuadtreeNode(node,center,bound):
    if (node == None):
        return
    if (not node[0]):
        rect = (int(center[0]-bound[0]/2.0),int(center[1]-bound[1]/2.0),int(bound[0]),int(bound[1]))
        pygame.draw.rect(screen,(255,0,0),rect,2)

        drawQuadtreeNode(node[1],center + np.array((bound[0]/4,-bound[1]/4)),bound/2)
        drawQuadtreeNode(node[2],center + np.array((-bound[0]/4,-bound[1]/4)),bound/2)
        drawQuadtreeNode(node[3],center + np.array((bound[0]/4,bound[1]/4)),bound/2)
        drawQuadtreeNode(node[4],center + np.array((-bound[0]/4,bound[1]/4)),bound/2)

def drawQuadtree():
    bound = np.array((BOUNDS[2]-BOUNDS[0],BOUNDS[3]-BOUNDS[1]))/(2.0 ** MAX_SUBDIVISIONS)
    for cell in quadtree:
        center = cell[0]
        rect = (int(center[0]-bound[0]/2),int(center[1]-bound[1]/2),int(bound[0]),int(bound[1]))
        pygame.draw.rect(screen,(255,200,200),rect,1)
    

while (run):
    
    deltaTime = time.time() - last
    last = time.time()
    
    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            run = False


    screen.fill((255,255,255))
    #generation
    generateQuadtree()
    
    #drawing (old)
    #bound = np.array((BOUNDS[2]-BOUNDS[0],BOUNDS[3]-BOUNDS[1]))
    #center = np.array((BOUNDS[0],BOUNDS[1]))+bound/2.0
    #drawQuadtreeNode(quadtree,center,bound)

    #drawing (new)
    if (DEBUG_DRAW):
        drawQuadtree()
    
    for i in range(len(boids)):
        nn = []
        for c in quadtree:
            for e in c:
                if (type(e) == int and e == i):
                    nn = c
                    break
            else:
                continue
            break
        nn.pop(0)
        updateBoid(i,deltaTime,nn)
    for boid in boids:
        drawBoid(boid)

    pygame.display.update()

pygame.quit()

    

    
