import numpy as np
from scipy import stats
# import tensorFlow as tf


class Params:
    default = {
        'Typeflag': 'c',
        'Stirke': 1,
        'Underlying': 1,
        'T2exp': 1,
        'Sigma': 0.1,
        'Price': 1,
        'Rf': 0, }

    def __init__(self, params=default):
        self.typeflag = params['Typeflag']
        self.strike = params['Stirke']
        self.underlying = params['Underlying']
        self.t2exp = params['T2exp']
        self.sigma = params['Sigma']
        self.price = params['Price']
        self.rf = params['Rf']
        self.d1 = (np.log(self.underlying / self.strike) + (self.rf + np.sqrt(self.sigma) / 2) * self.t2exp) / (self.sigma * np.sqrt(self.t2exp))
        self.d2 = (np.log(self.underlying / self.strike) + (self.rf - np.sqrt(self.sigma) / 2) * self.t2exp) / (self.sigma * np.sqrt(self.t2exp))


class Option(Params):
    def update_params(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None, renew=False):
        update = {
            'Typeflag': self.typeflag if typeflag is None else typeflag,
            'Stirke': self.strike if strike is None else strike,
            'Underlying': self.underlying if underlying is None else underlying,
            'T2exp': self.t2exp if t2exp is None else t2exp,
            'Sigma': self.sigma if sigma is None else sigma,
            'Price': self.price if price is None else price,
            'Rf': self.rf if rf is None else rf}

        if renew is True:
            self.__init__(update)
        return Params(update)

    def parity_formula(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        if params.typeflag == "c":
            result = params.price + params.strike * np.exp(-params.rf * params.t2exp) - params.underlying
        else:
            result = params.price - params.strike * np.exp(-params.rf * params.t2exp) + params.underlying
        return result


class BSMoption(Option):
    def get_price(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        if params.typeflag == "c":
            cdf_d1 = params.underlying * np.exp(-params.rf * params.t2exp) * stats.norm.cdf(params.d1)
            cdf_d2 = params.strike * np.exp(-params.rf * params.t2exp) * stats.norm.cdf(params.d2)
            price = cdf_d1 - cdf_d2
        elif params.typeflag == "p":
            cdf_d1 = params.underlying * np.exp(-params.rf * params.t2exp) * stats.norm.cdf(-params.d1)
            cdf_d2 = params.strike * np.exp(-params.rf * params.t2exp) * stats.norm.cdf(-params.d2)
            price = cdf_d2 - cdf_d1
        else:
            price = None
            print('This typeflag is wrong')
        return price

    def get_delta(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        if params.typeflag == "c":
            delta = np.exp(-params.rf * params.t2exp) * stats.norm.cdf(params.d1)
        elif params.typeflag == "p":
            delta = np.exp(-params.rf * params.t2exp) * (stats.norm.cdf(params.d1) - 1)
        else:
            delta = None
            print('This typeflag is wrong')
        return delta

    def get_gamma(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        gamma = np.exp(-params.rf * params.t2exp) * stats.norm.pdf(params.d1) / (params.underlying * params.sigma * np.sqrt(params.t2exp))
        return gamma

    def get_vega(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        vega = params.underlying * np.exp(-params.rf * params.t2exp) * stats.norm.pdf(params.d1) * np.sqrt(params.t2exp)
        return vega

    def get_theta(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        theta0 = -1 / 2 * params.underlying * np.exp(-params.rf * params.t2exp) * stats.norm.pdf(params.d1) * params.sigma / np.sqrt(params.t2exp)
        if params.typeflag == "c":
            theta = theta0 + params.rf * params.underlying * np.exp(-params.rf * params.t2exp) * stats.norm.cdf(+params.d1) \
                    - params.rf * params.strike * np.exp(-params.rf * params.t2exp) * stats.norm.cdf(+params.d2)
        elif params.typeflag == "p":
            theta = theta0 - params.rf * params.underlying * np.exp(-params.rf * params.t2exp) * stats.norm.cdf(-params.d1) \
                    + params.rf * params.strike * np.exp(-params.rf * params.t2exp) * stats.norm.cdf(-params.d2)
        else:
            theta = None
            print('This typeflag is wrong')
        return theta

    def get_rho(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        price = self.get_price(params.typeflag, params.strike, params.underlying, params.sigma, params.price, params.t2exp, params.rf)
        rho = -params.t2exp * price
        return rho

    def get_lamda(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        price = self.get_price(params.typeflag, params.strike, params.underlying, params.sigma, params.price, params.t2exp, params.rf)
        if params.typeflag == "c":
            lamda = np.exp(-params.rf * params.t2exp) * stats.norm.cdf(params.d1) * params.underlying / price
        elif params.typeflag == "p":
            lamda = np.exp(-params.rf * params.t2exp) * (stats.norm.cdf(params.d1) - 1) * params.underlying / price
        else:
            lamda = None
            print('This typeflag is wrong')
        return lamda

    def get_cofc(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        if params.typeflag == "c":
            cofc = params.t2exp * params.underlying * np.exp(-params.rf * params.t2exp) * stats.norm.cdf(params.d1)
        elif params.typeflag == "p":
            cofc = -params.t2exp * params.underlying * np.exp(-params.rf * params.t2exp) * stats.norm.cdf(-params.d1)
        else:
            cofc = None
            print('This typeflag is wrong')
        return cofc

    def get_volatility(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None, integral=False):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        set_sigma, top, floor = 5, 100, 0
        for i in range(100):
            bsm_price = self.get_price(params.typeflag, params.strike, params.underlying, set_sigma, params.price, params.t2exp, params.rf)
            diff = bsm_price - params.price
            if abs(diff) < 0.0001:
                return set_sigma
            if integral is False:
                if diff > 0:
                    floor = set_sigma
                else:
                    top = set_sigma
                set_sigma = (top + floor) / 2
            elif integral is True:
                vega = self.get_vega(params.typeflag, params.strike, params.underlying, set_sigma, params.price, params.t2exp, params.rf)
                set_sigma = set_sigma + diff / vega / 5
        return set_sigma


if __name__ == "__main__":
    params = Params()
    option = BSMoption()
    print(option.typeflag)
    option.update_params(underlying=1, renew=True)
    price = option.get_price()
    iv = option.get_volatility(price=price)
    print(price, iv)