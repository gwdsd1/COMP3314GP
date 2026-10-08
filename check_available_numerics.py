"""Independent checks for Gram PCA, complex Gabor convolution, and the original core numerical tests."""

import json
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d
from threadpoolctl import threadpool_limits

from csa import PCA, CSA
from gabor import kernel, features, CONFIG
from check_numerics import check


def main():
    result=check();rng=np.random.default_rng(731)
    x=rng.normal(size=(18,7,11))
    for center in [False,True]:
        a=PCA(8,center=center).fit(x);b=PCA(8,center=center,solver='gram').fit(x)
        np.testing.assert_allclose(a.basis@a.basis.T,b.basis@b.basis.T,atol=2e-12)
        np.testing.assert_allclose(a.inverse_transform(a.transform(x)),b.inverse_transform(b.transform(x)),atol=2e-12)
    result['checks'].append('Gram PCA projector and reconstruction agree with independent SVD')
    image=rng.normal(size=(1,9,7)).astype(np.float32)
    actual=features(image)
    maximum=0.
    for s in [0,2,4]:
        for o in [0,3,7]:
            k=kernel(CONFIG['wavelengths'][s],CONFIG['orientations'][o]);r=k.shape[0]//2
            padded=np.pad(image[0],((r,r),(r,r)),mode='reflect')
            expected=np.abs(convolve2d(padded,k,mode='valid'))
            maximum=max(maximum,float(np.max(np.abs(expected-actual[0,:,:,8*s+o]))))
            np.testing.assert_allclose(actual[0,:,:,8*s+o],expected,atol=2e-6,rtol=3e-5)
    constant=features(np.full((1,9,7),128,dtype=np.float32))
    assert float(np.max(constant))<1e-4
    np.testing.assert_allclose(features(image,step=2),actual[:,::2,::2],atol=0)
    result['checks'].extend(['Gabor FFT agrees with spatial complex convolution for 9 kernels',
                              'finite DC-free constant response and filter-before-downsample verified'])
    diagnostic=CSA((3,2),max_iter=10,stop_on_convergence=False).fit(x)
    assert len(diagnostic.history)==10
    result['checks'].append('10-iteration diagnostic does not alter default recognition stopping')
    result['gabor_max_absolute_error']=maximum
    path=Path(__file__).resolve().parent/'results/available_numerical_checks.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
