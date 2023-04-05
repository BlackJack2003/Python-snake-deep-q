import numpy as np
import random
import turtle,time

size= 40
sq2 = np.sqrt(2)
hlook = np.array([255,255])
flook = np.array([0,255])
blook=np.array([255,0])
blank=np.array([0,0])
bdc = True

if __name__ =="__main__":
    size=30
    fposy = [(5,5),(3,3),(6,6),(5,6),(6,7),(5,5)]

rf = 10/(2*size -1)

class InvalidInputError(Exception):
    print("Invalid Input val")
    
class player:
    def __init__(self,x=size//2,y=size//2):
        self.cx = x
        self.cy = y
        self.px =x
        self.py =y

class snake_board:
    #1 head,2 fruit,3 body,4 blank
    def setzone(self,x,y,t):
        if x>size-1 or y>size-1 or x<1 or y<1:
            print(f"out of bound de to x:{x},y:{y},t:{t}\n1 head,2 fruit,3 body,4 blank")
            quit()
        if t==1:
            i,j = 255,255
        elif t==2:
            i,j= 0,255
        elif t==3:
            i,j=255,0
        else:
            i,j=0,0
        for _ in range(-1,2):
            for __ in range(-1,2):
                self.board[x+_][y+__][0]=i
                self.board[x+_][y+__][1]=j

    def elpepe(self)->tuple:
        m = self.fpos[0]
        self.fpos.pop(0)
        if len(self.fpos)==1:
            for m in range(8):
                self.fpos.append((random.randint(2,size-2),random.randint(2,size-2)))
        return m

    def pepe(self):
        m,k = random.randint(2,size-2),random.randint(2,size-2)
        while self.board[m][k][0]!=0:
            m,k = random.randint(2,size-2),random.randint(2,size-2)
        return m,k

    def __init__(self,fpos=None):
        self.h = player()
        self.board = np.zeros((size,size,2),dtype=np.int16)
        self.segs = [self.h]
        self.setzone(self.h.cx,self.h.cy,0)
        if fpos==None:
            self.getfrp = lambda:self.pepe() 
        else:
            self.fpos = fpos
            self.getfrp = lambda: self.elpepe()
        self.fx,self.fy = self.getfrp()
        self.setzone(self.fx,self.fy,2)
        self.ps=abs(self.fx-self.h.cx) + abs(self.fy-self.h.cy)
        self.size=1
        self.pd = -1
        self.pmove=0
        #self.timestep=0
        
    def check_death(self)->bool:
        cx = self.h.cx
        cy = self.h.cy
        if self.h.cx < 1 or self.h.cx > size-2 or self.h.cy<1 or self.h.cy > size-2:
            return True
        for m in range(1,len(self.segs)):
            ch_x = abs(self.segs[m].cx -cx)
            ch_y = abs(self.segs[m].cy-cy)
            if (ch_x < 3 and ch_y < 3):
                return True
        return False
    
    def check_eat(self)->bool:
        ch_x = abs(self.fx -self.h.cx)
        ch_y = abs(self.fy-self.h.cy)    
        m = bool(ch_x < 3 and ch_y < 3)
        if m:
            self.setzone(self.fx,self.fy,4)
            self.fx,self.fy = self.getfrp()
            self.setzone(self.fx,self.fy,2)
            self.setzone(self.h.cx,self.h.cy,1)
            last = self.segs[-1]
            self.setzone(last.px,last.py,3)
            self.segs.append(player(last.px,last.py))
            self.size+=1
        return m
    
    #0 up,1 down, 2 left 3 right
    def move(self,dd:int):
        if dd==0:
            dirx=3
            diry=0
        elif dd==1:
            dirx=-3
            diry=0
        elif dd==2:
            dirx=0
            diry=3
        elif dd==3:
            dirx=0
            diry=-3
        else:
            raise InvalidInputError
        self.h.px=self.h.cx
        self.h.py=self.h.cy
        self.h.cx-=dirx
        self.h.cy-=diry
        d = self.check_death()
        if d:
            return True
        #check for border collision
        #trailing segments occupy the preceeding ones place
        """self.board[self.h.cx][self.h.cy][0]=255
        self.board[self.h.cx][self.h.cy][1]=255
        self.board[self.h.px][self.h.py][1]=0"""
        self.setzone(self.h.cx, self.h.cy,1)
        self.setzone(self.h.px, self.h.py,3)
        m=0
        for m in range(1,len(self.segs)):
            self.segs[m].px=self.segs[m].cx
            self.segs[m].py=self.segs[m].cy
            self.segs[m].cx = self.segs[m-1].px
            self.segs[m].cy = self.segs[m-1].py
        #set last ones position as free
        self.setzone(self.segs[-1].px,self.segs[-1].py,4)
        self.pmove=dd
        return False
    
    def step(self,action:int):
        d = self.move(action)
        eat = self.check_eat()
        #self.timestep+=1
        _ =abs(self.fx-self.h.cx) + abs(self.fy-self.h.cy)
        if eat==True:
            rew=2
        if d:
            rew=-4
        else:
            rew= 1 if _ < self.ps else -2
        self.ps = _
        return self.board,rew,d,self.size
    
    def reset(self,fpos:list=None):
        self.h = player()
        self.board = np.zeros((size,size,2),dtype=np.int16)
        m = np.ones(size,dtype=np.int16)
        self.segs = [self.h]
        self.setzone(self.h.cx,self.h.cy,0)
        if fpos==None:
            self.getfrp = lambda:self.pepe()
        else:
            self.fpos=fpos
            self.getfrp=lambda:self.elpepe()
        self.fx,self.fy = self.getfrp()
        self.setzone(self.fx,self.fy,2)
        self.ps=abs(self.fx-self.h.cx) + abs(self.fy-self.h.cy)
        self.size=1
        #self.timestep=0
        return self.board
    
    def render(self,actions,fpos):
        global bdc
        bdc = True
        bdd={True:"black",False:"green"}
        k = size*10
        wn = turtle.Screen()
        wn.tracer(0)
        self.reset(fpos)
        wn.title("Snake Game")
        wn.bgcolor("white")
        # the width and height can be put as user's choice
        wn.setup(width=max(500,size*21), height=max(500,size*21))
        head=turtle.Turtle()
        head.penup()
        head.setpos((self.h.cy*20)-k,(-20*self.h.cx)+k)
        head.shape('square')
        head.color('red')
        head.shapesize(3)
        segs=[head]
        food = turtle.Turtle()
        food.shape('square')
        food.color('blue')
        food.shapesize(3)
        food.penup()
        food.setpos((self.fy*20)-k,(self.fx*-20)+k)
        def add_seg(x,y):
            global bdc
            seg1 = turtle.Turtle()
            seg1.shape('square')
            seg1.color(bdd[bdc])
            bdc = not bdc
            seg1.shapesize(3)
            seg1.penup()
            seg1.goto(x,y)
            return seg1
        k_ = len(actions)
        for _ in range(len(actions)):
            a1,a2,a3,a4 = self.step(actions[_])
            food.setpos((self.fy*20)-k,(self.fx*-20)+k)
            if len(self.segs)>len(segs):
                segs.append(add_seg((self.segs[-1].cy*20)-k,(self.segs[-1].cx*-20)+k))
            for i,v in enumerate(self.segs):
                segs[i].setpos((v.cy*20)-k,(v.cx*-20)+k)
            print("Remianing:"+str(k_)+" Fpos:"+str(self.fy)+","+str(self.fx),",pos:",str(self.h.cx),",",str(self.h.cy),", reward:",str(a2))
            k_-=1
            time.sleep(0.5)
            wn.update()
        turtle.bye()
    
    def __str__(self)->str:
        tot = "\n    "
        for i in range(size):
            tot+=' '+str(i)
        tot+='\n     '
        for i in range(size):
            tot+=' #'
        tot+="\n"
        for i in range(size):
            r=str(i)+"# "
            for j in range(size):
                m = self.board[i][j]
                r+=' '
                if m[0]==0:
                    if m[1]==255:
                        r+='2'
                    else:
                        r+='0'
                else:
                    if m[1]==0:
                        r+='#'
                    else:
                        r+='H'
            tot+='\n'+r
        return tot+'\nSize: '+str(self.size)+'#'+str(self.h.cx)+'#'+str(self.h.cy)
    
if __name__ =="__main__":
    env = snake_board()
    env.reset(fposy)
    #0 up,1 down 2 left 3 right
    k =(0,2,0,2,0,2,0,2,0,2,0,2,2,0,2)
    for m in k:
        env.step(m)
        print(env)



    
