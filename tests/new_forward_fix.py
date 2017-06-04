from tests.test_base import *

def hard_fixing_r(param_optimizer, saved_r=None,
                  debug_jac=False, iterations=100, track_jacz_parts=False):
    tf.set_random_seed(0)
    np.random.seed(0)
    method = rf.ForwardHyperGradient
    iris, x, y, model, model_w, model_y, error, accuracy = iris_logistic_regression(
        param_optimizer.get_augmentation_multiplier())

    rho = tf.Variable([.1, .01], name='rho')
    tr_error = error \
               + rho[0]*tf.reduce_sum(model_w.tensor**2)\
               + rho[1]*tf.abs(tf.reduce_sum(model_w.tensor))

    eta = tf.Variable(.001, name='eta')
    dyn = param_optimizer.create(model_w, eta, loss=tr_error, _debug_jac_z=debug_jac)
    tr_sup = lambda s=None: {x: iris.train.data, y: iris.train.target}
    val_sup = lambda s=None: {x: iris.validation.data, y: iris.validation.target}

    hy_opt = rf.HyperOptimizer(dyn, {error: [eta
                                          # ]
        , rho]
    }, method=method, hyper_optimizer_class=None)

    track_tensors(*hy_opt.hyper_gradients.zs[0].var_list(),
                  *hy_opt.hyper_gradients.zs_dynamics[0].var_list())
    track_tensors(model_w.tensor)
    # track_tensors(dyn.global_step.var)
    # track_tensors(*hy_opt.hyper_gradients.zs[0].copy.var_list())

    all_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    tracked_names = [n for n in all_names if TRACK in n]
    tracked_tensors = {n: [] for n in tracked_names}

    r = hy_opt.hyper_gradients.zs[0].var_list()[0]
    r_saves = []

    # FIX R PART
    if saved_r:
        r_placeholder = tf.placeholder(tf.float32)
        r_assign = r.assign(r_placeholder)
    else: r_placeholder, r_assign = None, None

    # print('tracked names')
    # print(tracked_names)
    # print()
    with tf.Session().as_default() as ss:
        hy_opt.initialize()
        for it in range(iterations):
            r_saves.append(ss.run(r))
            if saved_r:
                if not np.array_equal(r_saves[-1], saved_r[it]):
                    print(it, 'divergence in r detected. Hard fixing it!')
                    ss.run(r_assign, feed_dict={r_placeholder: saved_r[it]})
                    print()

            if track_jacz_parts: [tracked_tensors[n].append(ss.run(n + ':0', feed_dict=tr_sup())) for n in tracked_names]

            # hy_opt.run(1, tr_sup, {error: val_sup}, _debug_no_hyper_update=True)
            hy_opt.hyper_gradients.step_forward(tr_sup)  # , {error: val_sup})  # , _debug_no_hyper_update=True)

        hy_opt.hyper_gradients.hyper_gradients({error: val_sup})
        hyp_gs = ss.run(hy_opt.hyper_gradients.hyper_gradient_vars)
        print(hyp_gs)

        if track_jacz_parts: [tracked_tensors[n].append(ss.run(n+':0', feed_dict=tr_sup())) for n in tracked_names]

        return hyp_gs, tracked_tensors, r_saves


def setUp():
    tf.reset_default_graph()


def trial_solve2():
    n_iters = 100
    trials = 100

    jac_z_quant_s = []
    res_f = []

    saved_r = None
    for j in range(trials):
        print(j)
        setUp()
        gh, tracked_tensors, saved_r = hard_fixing_r(param_optimizer=rf.AdamOptimizer, saved_r=None,
                                                        debug_jac=False, iterations=n_iters, track_jacz_parts=True)

        jac_z_quant_s.append(tracked_tensors)
        res_f.append(gh)

    print()
    [print('%.10f' % r) for r in res_f]

    tbr = []
    # for rs, z, gve in zip(res_f, res_z, res_gve):
    #     _ = np.linalg.norm(np.array(z))
    #     norm2 = np.linalg.norm(gve)
    #     tbr.append((rs[0], _, norm2))
    # print(tabulate.tabulate(tbr, floatfmt=".10f"))

    # assert_array_list_all_same(res_gve, raise_error=False, msg='gve: ')
    #
    # z0 = res_z[0]
    # dl0 = dpdl_s[0]
    jzq0 = jac_z_quant_s[0]

    [print(n) for n in list(jzq0.keys())]
    for k,  jzq in enumerate(jac_z_quant_s):

        print(k)
        # diff = assert_array_lists_same(z0, z, raise_error=False, msg='zs: ')
        #
        # if diff:
        for n in jzq0:
            if n in jzq:
                assert_array_lists_same(jzq0[n], jzq[n], raise_error=False, msg=n + ': ')
            else:
                print('misteri...', n, 'non ci sta')


        # assert_array_lists_same(dl0, dl, raise_error=False, msg='dl: ')


def test_z_merged_only():
    np.random.seed(0)
    dims = (10, 1)
    v1 = tf.Variable(tf.zeros(dims))
    v2 = tf.Variable(tf.zeros(dims))
    v3 = tf.Variable(tf.zeros(dims))

    iters = 100

    seq1 = [np.random.randn(*dims) for _ in range(1)]
    seq2 = [np.random.randn(*dims) for _ in range(1)]
    seq3 = [np.random.randn(*dims) for _ in range(1)]
    seq = [seq1[-1], seq2[-1], seq3[-1]]

    vrs = [v1, v2, v3]
    z = rf.ZMergedMatrix(vrs)
    z_pls = [tf.placeholder(tf.float32, shape=dims) for _ in range(len(vrs))]
    z_assign = z.assign(z_pls)

    res = []
    print('begin')
    with tf.Session().as_default() as ss:
        tf.global_variables_initializer().run()
        for j in range(iters):

            res.append(z.tensor.eval())
            ss.run(z_assign, {p: v for p, v in zip(z_pls, seq)})
    # print('a', end='')
    return res

def tst_z_only():
    runs = 100
    resss = [test_z_merged_only() for _ in range(runs)]

    r0 = resss[0]
    for k, r in enumerate(resss):
        # print(k)
        assert_array_lists_same(r0, r, raise_error=False, msg='trial %d: : ' % k)


if __name__ == '__main__':
    trial_solve2()
    # tst_z_only()
    pass