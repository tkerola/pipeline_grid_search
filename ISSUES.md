
Issues
======

PipelineGridSearchCV does not work well with FeatureUnion.

Reason
------

Pipeline([
  PCA(),
  Normalizer(),
  SVC(),
])

vs.

Pipeline([
  PCA pca (ncomp in [3,5,7]),
  FeatureUnion fu ([
  PCA pca1 (ncomp in [1,2,3]),
  PCA pca2 (ncomp in [4,5,6]),
  ]),
  SVC svm (C in [10,100]),
])

Param names:
```
pca__ncomp
fu__pca1__ncomp
fu__pca2__ncomp
svm__C
```

Don't recompute fu if `fu__pca1__ncomp` and `fu__pca2__ncomp` are unchanged.
If `fu__pca1__ncomp` is unchanged, but `fu__pca2__ncomp` is changed:
 * Keep result from pca1, but call fit on pca2.
 * How does this work with `clone_steps`?
   * `_get_start_state` must be changed so that
     `clone_steps` is not called on both fu(left,right).
     It should be called only on fu(right) in this case.
     * Actually, this is not needed. Adding polymorphic versions of
       `clone`, `fit` and `fit_transform` should be enough.
   * `_get_start_state` is used in
     * `clone_steps`
     * `_pre_transform`
     * `score`
   * Need to wrap each call to `fit` so that only the appropriate
     left or right side of fu is called, and the cached transform
     result is used for the other side.
     * The same for `clone` and `transform`
     * `class GridSearchCVFeatureUnion`
   * Problem:
     * Even if `GridSearchCVFeatureUnion` would avoid recomputation of
       any left/right part, this will only work as expected if
parameters arrive in an optimal order. DFS order is optimal for a normal
Pipeline, but what order will be suitable for a FeatureUnion?
     * Change right params in DFS order. This will reuse the cached
       results from the left side.
     * Then change on param on the left side (in DFS order) and once
       again change all params on the right side in DFS order.
     * This scheme will try all combinations of the left/right params
       and avoid recomputaion of the left side most of the time.
       * Could we make it possible to switch the left/right side
         to avoid recomputation on the side which takes the most time?

```
class GridSearchCVFeatureUnion:
  def clone(params):
    if left side changed:
      left_est = clone(left_est)
    if right side changed:
      right_est = clone(right_est)
  def fit(X, params):
    if left side changed:
      X_left = left_est.fit(X)
      cache[left] = X_left
    else:
      X_left = cache[left]
    if right side changed:
      X_right = right_est.fit(X)
      cache[right] = X_right
    else:
      X_right = cache[right]
  def fit_transform(X):
    pass
```
```
fu__pca1__ncomp in [1,2,3] # left
fu__pca2__ncomp in [4,5,6] # right

DFS order
left 1, right 4
left 1, right 5
left 1, right 6
left 2, right 4
left 2, right 5
left 2, right 6
left 3, right 4
left 3, right 5
left 3, right 6

"Gray code" order?
left 1, right 4
left 1, right 5
left 2, right 5
left 2, right 6
left 3, right 6
left 3, right 4
left 2, right 4
left 2, right 5
left 1, right 5
left 1, right 6

left -> right -> svm (etc.)

3 calls to left, 3*3 calls to right.
Ideally, we would want 3 calls to both left and right.
This should be possible, as left/right are completely independent of
eachother.
The problems is that this setup only caches the most recent call to
an estimator.
So with this setup, solving the problem might not be possible.
We would need to cache each possible param config in the right side
(this is the approach taken in the `file_based` branch).
The problem with storing anything but the last computed value
is that we would need memory exponential in the number of parameters
to store all the possible combinations of parmeters in the pipeline
on the right side.
For the left side, we don't need to store anything except the
previously computed parameter, if the params arrive in DFS
order.
```

FeatureUnion has two parallel parts (left, right).

Left                Right
```
parts = [
    PCA(),
    Normalizer(),
    SVC()
    ]

cv_params = [
    ('PCA__n_components', [3,5,7]),
    ('Normalizer__norm', ['l2']),
    ('SVC__C', [1.,10.,100.,1000.]),
]
```
