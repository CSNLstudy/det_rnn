import copy

class scheduler_none():
    def __init__(self, hp):
        self.hp = copy.deepcopy(hp)

    def get_parms(self):
        train_params = {
            'loss_mse_dec': self.hp['loss_mse_dec'],
            'loss_mse_est': self.hp['loss_mse_est'],
            'loss_ce_dec': self.hp['loss_ce_dec'],
            'loss_ce_est': self.hp['loss_ce_est'],
            'loss_spike': self.hp['loss_spike'],
            'loss_L1': self.hp['loss_L1'],
            'loss_L2': self.hp['loss_L2'],
            'dropout': self.hp['dropout'],
            'learning_rate': self.hp['learning_rate'],
            'clip_max_grad_val': self.hp['clip_max_grad_val']
        }

        return train_params

    def update(self, t, est_perf, dec_perf):
        # do nothing
        a = t + 1


class scheduler_estimFirst():
    def __init__(self, hp):
        self.hp = copy.deepcopy(hp)
        self.switch = False  # switch to est and dec.
        self.decay = 0.99
        self.nbad = 0
        self.baseLR = hp['learning_rate']

    def get_params(self):
        train_params = {
            'loss_mse_dec': self.hp['loss_mse_dec'],
            'loss_mse_est': self.hp['loss_mse_est'],
            'loss_ce_dec': self.hp['loss_ce_dec'],
            'loss_ce_est': self.hp['loss_ce_est'],
            'loss_spike': self.hp['loss_spike'],
            'loss_L1': self.hp['loss_L1'],
            'loss_L2': self.hp['loss_L2'],
            'dropout': self.hp['dropout'],
            'learning_rate': self.hp['learning_rate'],
            'clip_max_grad_val': self.hp['clip_max_grad_val']
        }

        if not self.switch:  # set decision weights to 0
            train_params['loss_mse_dec'] = 0
            train_params['loss_ce_dec'] = 0
            train_params['dropout'] = 0

        return train_params

    def update(self, t, est_perf, dec_perf):
        # learn estimation first, and makes sure est is good
        if self.switch == False:
            if est_perf > 0.95:  # estimation and decision mode
                self.switch = True
                self.nbad = 0
            if est_perf < 0.7:
                self.nbad += 1

        if self.switch == True:
            if est_perf < 0.8:  # If estimation drops too low, switch back to estimation only
                self.switch = False
                self.nbad += 1

            if dec_perf < 0.6:  # bad performance
                self.nbad += 1

        if est_perf > 0.75 and dec_perf > 0.75:  # try fine tuning, lower learning rate
            self.nbad = 0
            self.hp['learning_rate'] = self.hp['learning_rate'] * self.decay

        if self.nbad > 10:  # try increasing learning rate
            # may be stuck at local minima.
            # note that ADAM optimizer by default normalizes learning by gradient norm.
            self.hp['learning_rate'] = self.hp['learning_rate'] / self.decay


class scheduler_separate():
    def __init__(self, hp):
        self.hp = copy.deepcopy(hp)
        self.switch = False  # switch to est and dec.
        self.decay = 0.99
        self.nbad = 0
        self.baseLR = hp['learning_rate']

    def get_params(self):
        train_params = {
            'loss_mse_dec': self.hp['loss_mse_dec'],
            'loss_mse_est': self.hp['loss_mse_est'],
            'loss_ce_dec': self.hp['loss_ce_dec'],
            'loss_ce_est': self.hp['loss_ce_est'],
            'loss_spike': self.hp['loss_spike'],
            'loss_L1': self.hp['loss_L1'],
            'loss_L2': self.hp['loss_L2'],
            'dropout': self.hp['dropout'],
            'learning_rate': self.hp['learning_rate'],
            'clip_max_grad_val': self.hp['clip_max_grad_val']
        }

        if not self.switch:  # set decision weights to 0
            train_params['loss_mse_dec'] = 0
            train_params['loss_ce_dec'] = 0
            train_params['dropout'] = 0

        if self.switch:
            train_params['loss_mse_est'] = 0
            train_params['loss_ce_est'] = 0
            train_params['dropout'] = 0

        return train_params

    def update(self, t, est_perf, dec_perf):
        # learn estimation first, and makes sure est is good
        if self.switch == False:
            if est_perf > 0.95:  # estimation and decision mode
                self.switch = True
                self.nbad = 0
            if est_perf < 0.7:
                self.nbad += 1

        if self.switch == True:
            if est_perf < 0.8:  # If estimation drops too low, switch back to estimation only
                self.switch = False
                self.nbad += 1

            if dec_perf < 0.6:  # bad performance
                self.nbad += 1

        if est_perf > 0.75 and dec_perf > 0.75:  # try fine tuning, lower learning rate
            self.nbad = 0
            self.hp['learning_rate'] = self.hp['learning_rate'] * self.decay

        if self.nbad > 10:  # try increasing learning rate
            # may be stuck at local minima.
            # note that ADAM optimizer by default normalizes learning by gradient norm.
            self.hp['learning_rate'] = self.hp['learning_rate'] / self.decay


class scheduler_timeconst():
    def __init__(self, hp):
        self.hp = copy.deepcopy(hp)
        self.taumin = hp['tau_min']
        self.taumax = hp['tau_max']
        self.switch = 0
        self.nbad = 0
        self.baseLR = hp['learning_rate']

    def get_params(self):
        train_params = {
            'loss_mse_dec': self.hp['loss_mse_dec'],
            'loss_mse_est': self.hp['loss_mse_est'],
            'loss_ce_dec': self.hp['loss_ce_dec'],
            'loss_ce_est': self.hp['loss_ce_est'],
            'loss_spike': self.hp['loss_spike'],
            'loss_L1': self.hp['loss_L1'],
            'loss_L2': self.hp['loss_L2'],
            'dropout': self.hp['dropout'],
            'learning_rate': self.hp['learning_rate'],
            'clip_max_grad_val': self.hp['clip_max_grad_val']
        }

        # try estimation only
        train_params['loss_mse_dec'] = 0
        train_params['loss_ce_dec'] = 0
        train_params['dropout'] = 0

        return train_params

    def update(self, t, est_perf, dec_perf):
        # learn estimation first, and makes sure est is good

        if est_perf > 0.95:  # estimation and decision mode
            self.nbad = 0
            self.taumax = self.taumax * 0.9
        else:
            self.nbad += 1

        if self.nbad > 20:  # increase tau max
            self.nbad = 0
            self.taumax = self.taumax * 1.1

        if self.taumax < self.taumin:
            self.taumax = self.taumin
