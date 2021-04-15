function wf = train_ReCF(params, xf, yf, s, xf_p, wf_p)

Sxy = xf .* conj(yf);
Sxx = xf .* conj(xf);
Sxx_p = xf_p .* conj(xf_p);

% feature size
sz = size(xf);
N = sz(1) * sz(2);

% initialize hf
hf = zeros(sz);

% initialize lagrangian multiplier
zetaf = zeros(sz);

% ReCF parameters
gamma_I = params.gamma_I; % Parameter on Inferred response regularization
gamma_H = params.gamma_H; % Parameter on historical response regularization

% ADMM parameters
mu = params.mu;
beta = params.beta;
mu_max = params.mu_max;

% ADMM iterations
iter = 1;
while (iter <= params.admm_iterations)
    wf = (Sxy + (wf_p .* (gamma_I * Sxx + gamma_H * Sxx_p)) + mu * hf - zetaf) ./...
         ((1 + gamma_I) * Sxx + gamma_H * Sxx_p + mu);
    hf = fft2(ifft2(mu * wf + zetaf, 'symmetric') ./ (1/N * s.^2 + mu));
    zetaf = zetaf + mu * (wf - hf);
    mu = min(mu_max, beta * mu);
    iter = iter + 1;
end

