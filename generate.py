import sys
import numpy as np

sys.path.append("/home/wxy/Program/veto_eff/build/")  # 添加编译生成的模块的路径
import mu_sim
# from deprecate.muon_Gen import energy


class Generator:
    def __init__(self):
        self.x_range = [-1000, 1000]
        self.y_range = [-100, 100]
        self.z = [10, 11] 
#        pass

    def get_theta(self):
        rand_num = np.random.uniform(0, 1)
        theta_value = mu_sim.MuSimPrimaryGeneratorAction.inverseCDFTheta(rand_num)
        return theta_value 
    

    def get_phi(self):
        phi=np.random.uniform(0,2*np.pi)
        return phi

    
    def get_energy(self):
        # 现在还是固定的能量
        # energy = mu_sim.MuSimPrimaryGeneratorAction.sample_energy_from_gaisser_tang(0.5)
        pass

    def get_position(self):
        # TODO: 返回一个随机的位置
        x = np.random.uniform(*self.x_range)
        y = np.random.uniform(*self.y_range)
        z = np.random.uniform(*self.z)
        return x, y, z

    def get_generator(self):
        x, y, z = self.get_position()
        energy = self.get_energy()
        theta = self.get_theta()
        return (x, y, z), energy, theta

