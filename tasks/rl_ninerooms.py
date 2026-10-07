# Copyright (c) 2017-2021 Katori Lab. All Rights Reserved
# rl_ninerooms 強化学習９つの部屋タスク
import numpy as np
import pygame
from pygame.locals import *
import sys
verbose=0

def get_intersection(rx,ry,sx,sy,x1,y1,x2,y2):
    """
    交点を取得する
    ①ロボットの中心とセンサの座標の直線
    ②壁や障害物の直線
    ①と②の交点を取得し、交点を返す
    (rx,ry):ロボの中心座標
    (sx,sy):伸ばしているセンサの向きの座標
    (x1,y1)(x2,y2):壁の座標
    (cx,cy):壁とセンサの交点
    """
    fx,fy=10000,10000

    a0 = sx - rx
    b0 = sy - ry
    a2 = x2 - x1
    b2 = y2 - y1

    d = a0*b2 - a2*b0
    if d == 0:#２直線が並行のとき（or重なっているとき)
        return False,fx,fy

    sn = b2 * (x1-rx) - a2 * (y1-ry)
    s = sn/d#①の媒介変数
    #cx = rx + a0*sn/d#交点
    #cy = ry + b0*sn/d

    tn = b0 * (x1-rx) - a0 * (y1-ry)
    t = tn/d#②の媒介変数
    cx = x1 + a2*tn/d#交点
    cy = y1 + b2*tn/d

    if 0 <= t <= 1:#交点が線分（壁）の中にあるか
        if 0<s:#ベクトルがセンサ座標の方向に伸びているとき。sは媒介変数
            return True,cx,cy

    return False,fx,fy

def distance(a,b,c,d):
    #距離を測る
    x = np.array([a,b])
    y = np.array([c,d])
    u = y-x
    dis = np.linalg.norm(u)
    return dis

def get_nearest_intersection( rx, ry, sx,sy ,walls):
    """
    各距離センサーについて、すべての壁との距離を測り、最も近くの壁との距離を返す。
    rx,ry: ロボット位置の座標
    theta: センサーの向き
    walls: 壁の集合
    """
    cx = rx#エラーの原因
    cy = ry#
    cd = 10000
    cross = False
    for j in range(len(walls)):# すべての壁について
        wall = walls[j]
        for i in range(len(wall)):
            if i < len(wall)-1:
                wx0,wy0 = wall[i  ][0],wall[i  ][1]
                wx1,wy1 = wall[i+1][0],wall[i+1][1]
            else:
                wx0,wy0 = wall[i][0],wall[i][1]
                wx1,wy1 = wall[0][0],wall[0][1]
            cross,bx,by = get_intersection(rx,ry,sx,sy,wx0,wy0,wx1,wy1)
            if cross:
                if distance(rx,ry,bx,by)<cd:
                    cx=bx
                    cy=by
                    cd=distance(rx,ry,cx,cy)
    return cd,cx,cy

def get_distance_to_wall( rx, ry, theta, degree, walls):
    """
    距離センサーで壁までの距離を測る。
    rx,ry,theta:ロボットの座標と向き
    degree: センサーの向き
    walls: 壁の集合
    poi_d: 交点までの距離
    poi_p: 交点の座標
    """
    poi_d = [] # points of intersection (distance)
    poi_p = [] # points of intersection (position)
    for i in range(len(degree)):# すべての距離センサーについて
        angle = theta + np.deg2rad(degree[i])
        sx = rx + np.cos(angle) # 伸ばしているセンサの向きの座標
        sy = ry - np.sin(angle)
        pp=[0,0]
        pd,pp[0],pp[1] = get_nearest_intersection(rx,ry,sx,sy,walls)
        poi_d.append(pd)
        poi_p.append(pp)
    return poi_d,poi_p

def check_lines_intersection(p1x,p1y,p2x,p2y,p3x,p3y,p4x,p4y):
    """
    p1(p1x,p1y)とp2(p2x,p2y)を結ぶ線分と、p3(p3x,p3y)とp4(p4x,p4y)を結ぶ線分が交差するときTrueを返す。
    """
    tc1 = (p1x - p2x) * (p3y - p1y) + (p1y - p2y) * (p1x - p3x)
    tc2 = (p1x - p2x) * (p4y - p1y) + (p1y - p2y) * (p1x - p4x)
    td1 = (p3x - p4x) * (p1y - p3y) + (p3y - p4y) * (p3x - p1x)
    td2 = (p3x - p4x) * (p2y - p3y) + (p3y - p4y) * (p3x - p2x)
    return tc1*tc2<0 and td1*td2<0

def check_objects_intersection(px0,py0,px1,py1,objects):
    """
    オブジェクト（壁など）との交差をチェックする。
    (px0,py0): 前の時間ステップにおけるロボット位置
    (px1,py1): 現在の時間ステップにおけるロボット位置
    objects: オブジェクトの集合
    cross: 交差した場合は True を返す。交差したオブジェクトのインデックスも返す。
    """
    cross = False
    i_cross = -1
    for j in range(len(objects)):
        object = objects[j]
        for i in range(len(object)):
            if i < len(object)-1:
                wx0,wy0 = object[i  ][0],object[i  ][1]
                wx1,wy1 = object[i+1][0],object[i+1][1]
            else:
                wx0,wy0 = object[i][0],object[i][1]
                wx1,wy1 = object[0][0],object[0][1]

            #print(i,len(object),px0,py0,px1,py1,w0x,w0y,w1x,w1y)
            if check_lines_intersection(px0,py0,px1,py1,wx0,wy0,wx1,wy1):
                cross = True
                i_cross = j
                return cross, i_cross
                #px = px0
                #py = py0
    return cross, i_cross
    #return px,py,cross

def calcurate_wheel(action):
    """
    離散値で与えられる行動命令をロボットの動作速度に変換する。
    action: 0,1,2のいずれかの値をとる行動命令
    v:ロボットの速度
    omega:ロボットの回転角速度
    """
    d = 25#ロボットの中心からwheelまでの距離
    wheel_radius = 5#車輪の半径

    if action == 0:
        phiR = 8#回転角度Right
        phiL = 4#回転角度Left
    if action == 1:#Right
        phiR = 4#回転角度Right
        phiL = 8#回転角度Left
    if action == 2:
        phiR = 8#回転角度Right
        phiL = 8#回転角度Left
    #if action == 3:
    #    phiR = -8#回転角度Right
    #    phiL = -8#回転角度Left

    vR = wheel_radius * phiR#車輪の速度Right
    vL = wheel_radius * phiL#車輪の速度Left
    omega = (vR - vL)/ (2 * d) # ロボットの回転角速度
    v = (vR + vL)/ 2#ロボットの中心速度
    return v,omega

class RobotRoom:
    def __init__(self,c):

        self.plot_orbit = c.plot_orbit # 軌道の保存の有無
        self.reward = 0
        self.goal = 0
        self.t = 0
        self.state = None
        self.clash = 0
        self.max_timesteps = c.max_timesteps

        ### ロボットの設定
        self.deg = [0,45,90,135,180,-45,-90,-135]#センサの角度
        self.dt = 0.3#[s]
        self.cp = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]#センサがあたっている、壁・障害物の座標
        self.cd=[-1,-1,-1,-1,-1,-1,-1,-1]#ロボットの中心と、センサの座標の距離(距離センサの値)

        ### 壁の設定 9つの部屋
        self.walls = []
        self.walls.append([[50,50],[950,50],[950,950],[50,950]])#壁
        self.walls.append([[50,50],[130,50],[210,50],[210,130],[130,130],[130,210],[50,210],[50,130],[50,50]])
        self.walls.append([[610,50],[690,50],[690,130],[610,130],[610,50]])
        self.walls.append([[850,50],[950,50],[950,150],[850,150],[850,50]])
        self.walls.append([[320,200],[380,200],[380,320],[450,320],[450,380],[380,380],[380,390],[320,390],[320,380],[250,380],[250,320],[320,320]])
        self.walls.append([[620,240],[680,240],[680,320],[780,320],[780,380],[680,380],[680,430],[620,430],[620,380],[620,320]])
        self.walls.append([[890,320],[950,320],[950,380],[890,380],[890,320]])
        self.walls.append([[50,380],[110,380],[110,620],[160,620],[160,680],[110,680],[50,680],[50,620]])
        self.walls.append([[320,590],[380,590],[380,620],[380,680],[380,740],[320,740],[320,680],[320,620]])
        self.walls.append([[620,620],[680,620],[830,620],[830,680],[680,680],[680,740],[620,740],[620,680],[520,680],[520,620],[620,620]])
        self.walls.append([[50,850],[150,850],[150,950],[50,950],[50,850]])
        self.walls.append([[620,840],[680,840],[680,890],[950,890],[950,950],[680,950],[620,950],[620,890]])
        self.walls.append([[320,850],[380,850],[380,870],[320,870],[320,850]])  # スタート付近に壁追加

        ### スタート地点の設定　
        self.i_start = 2 # スタート地点のインデックス
        self.starts = []
        self.starts.append([np.deg2rad(30),200,200])#A 角度、ｘ座標、ｙ座標
        self.starts.append([np.deg2rad(180),800,200])#C
        self.starts.append([np.deg2rad(45),200,800])#G
        self.starts.append([np.deg2rad(90),800,800])#I

        ### ゴール地点の設定
        self.i_goal = 0 # ゴール地点のインデックス
        self.goals = []
        self.goals.append([800,200])

        self.goals_c = []
        self.goals_c.append([[610,50],[610,130],[680,130],[680,320],[950,320],[950,150],[850,150],[850,50]]) ###ゴール

        ### robot軌道plot準備
        self.x_oribit = []
        self.y_oribit = []

        ### 領域の設定、ゴールの領域など面積を持つ範囲の指定に使う。
        self.areas = []

        ### pygame
        pygame.init()#Pygameの初期化
        self.im = pygame.image.load("./tasks/robot.jpg")#robot画像読み込み
        self.im = pygame.transform.rotozoom(self.im, -90, 0.5)#画像の縮小と回転（,角度,倍率）
        self.clock = pygame.time.Clock()
        self.viewer = 0

        SCR_WIDTH,SCR_HEIGHT = 1400,1000#画面の大きさ
        if c.is_render:
            self.screen = pygame.display.set_mode((SCR_WIDTH, SCR_HEIGHT))#画面を生成

    def reset(self):#状態を初期化し、初期の観測値を返す
        self.theta,self.px,self.py = self.starts[self.i_start]
        self.reward = 0
        self.state = None
        self.clash = 0
        self.goal = 0
        self.t = 0
        done = False

        self.x_oribit,self.y_oribit = [],[]

        return np.array(self.state), self.reward, done, {}

    def step(self, action):#action に応じて環境を変化させ、結果(state,reward)を返す
        ### ロボットの動作
        self.clash = 0
        self.goal = 0
        v,omega = calcurate_wheel(action)
        vx =  v*np.cos(self.theta)
        vy = -v*np.sin(self.theta)
        px0 = 1.0*self.px
        py0 = 1.0*self.py
        self.px = self.px + vx*self.dt#座標更新x
        self.py = self.py + vy*self.dt#座標更新y
        self.theta = self.theta + omega*self.dt#角度の更新

        self.clash,i_clash = check_objects_intersection(px0,py0,self.px,self.py,self.walls)
        self.cd, self.cp = get_distance_to_wall( self.px, self.py, self.theta, self.deg, self.walls)

        ### ゴール判定
        self.goal, i_goal= check_objects_intersection(px0,py0,self.px,self.py,self.goals_c)

        ### Reward
        i = 0 #np.argmax(object_area
        dis = distance(self.px,self.py,self.goals[i][0],self.goals[i][1])
        self.reward = 1.0/(1+dis/10)

        if self.clash == 1:
            self.reward = -1.5
        
        if self.goal == 1:
            self.reward += 5

        ### Done
        done = False
        if self.clash == 1:
            done = True
            if verbose: print("clash",i_clash)

        if self.goal == 1:
            done = True
            if verbose: print("goal")

        if self.t == self.max_timesteps-1:
            done = True

        ### State
        self.state = np.exp(-np.array(self.cd)/100)#壁までの距離を(0,1)区間の値に変換
        #self.state = self.cd
        self.t += 1

        if self.plot_orbit == True : #軌道の保存
            self.x_oribit.append(self.px)
            self.y_oribit.append(self.py)

        return self.state, self.reward, done, {}

    def render(self):#環境を可視化する
        pygame.display.set_caption("robot_simulation")#タイトルバーに表示する文字

        self.screen.fill((255,255,255))#画面を白に塗りつぶす
        ### Walls
        for i in range(len(self.walls)):#壁を描く
            wall = self.walls[i]
            for j in range(len(wall)):
                if j==len(wall)-1:
                    pygame.draw.line(self.screen,(0,0,0),(wall[j][0],wall[j][1]),(wall[0][0],wall[0][1]),4)
                else:
                    pygame.draw.line(self.screen,(0,0,0),(wall[j][0],wall[j][1]),(wall[j+1][0],wall[j+1][1]),4)
        ### ロボット
        theta_deg = np.rad2deg(self.theta)
        new_im = pygame.transform.rotate(self.im,theta_deg)#ロボットの角度
        rect = new_im.get_rect()
        rect.center = (self.px,self.py)# TODO 中心位置を設定
        self.screen.blit(new_im,rect)#画面に描写

        ### 文字列
        sysfont = pygame.font.SysFont(None,40)
        (text_x,text_y) = (1200,0)
        text_dy = 30

        ### センサ値とセンサの線の表示
        for g in range(len(self.deg)):
            pygame.draw.aaline(self.screen,(255,0,0),(self.px,self.py),(self.cp[g][0],self.cp[g][1]),2)#一番近い点との線を描く###
            dis=self.cd[g]
            messagese = sysfont.render("%4d: %6.2f" % (self.deg[g],round(dis,2)),True,(0,0,0))
            self.screen.blit(messagese,(text_x,text_y))
            text_y += text_dy

        messagese = sysfont.render("reward: %6.2f" % (self.reward),True,(0,0,0))
        self.screen.blit(messagese,(text_x,text_y))

        pygame.display.update()#画面更新
        self.clock.tick(30)#30fps

    def close(self):#環境を閉じて後処理をする
        if self.viewer: self.viewer.close()

    def handle_event(self):
        for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    print("quit...")
                    #if plot == 1:
                    #    plot()
                    sys.exit()
