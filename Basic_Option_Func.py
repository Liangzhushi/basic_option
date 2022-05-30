import numpy as np
from scipy import stats, integrate


class Params:
    default = {
        'Typeflag': 'c',
        'Strike': 1,
        'Underlying': 1,
        'T2exp': 1,
        'Sigma': 0.1,
        'Price': 1,
        'Rf': 0, }

    def __init__(self, params=default):
        self.typeflag = params.get('Typeflag')
        self.strike = params.get('Strike')
        self.underlying = params.get('Underlying')
        self.t2exp = params.get('T2exp')
        self.sigma = params.get('Sigma')
        self.price = params.get('Price')
        self.rf = params.get('Rf') if isinstance(params.get('Rf'), type(None)) else 0
        self.discount = np.exp(-self.rf * self.t2exp) if isinstance(self.t2exp, type(None)) else 1
        self.cp = 1 if self.typeflag == "c" else -1
        try:
            self.d1 = (np.log(self.underlying / self.strike) + (self.rf + pow(self.sigma, 2) / 2) * self.t2exp) / (self.sigma * np.sqrt(self.t2exp))
            self.d2 = (np.log(self.underlying / self.strike) + (self.rf - pow(self.sigma, 2) / 2) * self.t2exp) / (self.sigma * np.sqrt(self.t2exp))
        except Exception as e:
            self.d1, self.d2 = None, None


class Option(Params):
    def update_params(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None, renew=False):
        update = {
            'Typeflag': self.typeflag if typeflag is None else typeflag,
            'Strike': self.strike if strike is None else strike,
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
        result0 = params.strike * params.discount - params.underlying
        result = params.price + params.cp * result0
        return result


class BSMoption(Option):
    def get_price(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        cdf_d1 = params.underlying * params.discount * stats.norm.cdf(params.cp * params.d1)
        cdf_d2 = params.strike * params.discount * stats.norm.cdf(params.cp * params.d2)
        price = params.cp*(cdf_d1 - cdf_d2)
        return price

    def get_delta(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        delta = params.cp * params.discount * stats.norm.cdf(params.cp*params.d1)
        return delta

    def get_gamma(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        gamma = params.discount * stats.norm.pdf(params.d1) / (params.underlying * params.sigma * np.sqrt(params.t2exp))
        return gamma

    def get_vega(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        vega = params.underlying * params.discount * stats.norm.pdf(params.d1) * np.sqrt(params.t2exp)
        return vega

    def get_theta(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        theta0 = -1 / 2 * params.underlying * params.discount * stats.norm.pdf(params.d1) * params.sigma / np.sqrt(params.t2exp)
        theta = theta0 - params.cp * params.rf * params.strike * params.discount * stats.norm.cdf(params.cp*params.d2)
        return theta

    def f(self, t2exp=None, typeflag=None, strike=None, underlying=None, sigma=None, price=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        theta0 = -1 / 2 * params.underlying * params.discount * stats.norm.pdf(params.d1) * params.sigma / np.sqrt(params.t2exp)
        theta = theta0 - params.cp * params.rf * params.strike * params.discount * stats.norm.cdf(params.cp*params.d2)
        return theta

    def get_theta_quad(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        theta_quad, err = integrate.quad(self.f, 0, params.t2exp, args=(params.typeflag, params.strike, params.underlying, params.sigma, params.price, params.rf))
        return theta_quad

    def get_rho(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        price = self.get_price(params.typeflag, params.strike, params.underlying, params.sigma, params.price, params.t2exp, params.rf)
        rho = -params.t2exp * price
        return rho

    def get_lamda(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        price = self.get_price(params.typeflag, params.strike, params.underlying, params.sigma, params.price, params.t2exp, params.rf)
        lamda = params.discount * stats.norm.cdf(params.cp*params.d1) * params.underlying / price
        return lamda

    def get_cofc(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        cofc = params.cp * params.t2exp * params.underlying * params.discount * stats.norm.cdf(params.cp*params.d1)
        return cofc

    def get_volatility(self, typeflag=None, strike=None, underlying=None, sigma=None, price=None, t2exp=None, rf=None, integral=True):
        params = self.update_params(typeflag, strike, underlying, sigma, price, t2exp, rf)
        set_sigma, top, floor = 5, 100, 0
        for i in range(100000):
            bsm_price = self.get_price(params.typeflag, params.strike, params.underlying, set_sigma, params.price, params.t2exp, params.rf)
            diff = params.price / bsm_price - 1  
            if abs(diff) < 0.0001:  # 相较价格差异万分之一，而非绝对值，防止精度出现问题
                return set_sigma
            if integral is False:
                if diff > 0:
                    floor = set_sigma
                else:
                    top = set_sigma
                set_sigma = (top + floor) / 2
            elif integral is True:
                vega = self.get_vega(params.typeflag, params.strike, params.underlying, set_sigma, params.price, params.t2exp, params.rf)
                set_sigma = set_sigma + diff / vega
        return set_sigma


if __name__ == "__main__":
    test = {
        'Typeflag': 'c',
        'Strike': 40000,
        'Underlying': 38533,
        'T2exp': 7.5/365,
        'Sigma': 0.744,
        'Price': 1043,
        'Rf': 0, }
    params = Params(test)
    option = BSMoption(test)
    print(option.price, option.sigma, option.t2exp)
    # option.update_params(underlying=1, renew=True)
    price = option.get_price()
    iv = option.get_volatility()
    print(price, iv)
    theta = option.get_theta()
    print(theta)
    theta_quad = option.get_theta_quad()
    print(theta_quad)
