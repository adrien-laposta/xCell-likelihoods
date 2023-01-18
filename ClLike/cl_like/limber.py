"""
Theory class that computes the angular power spectra. It uses the CCL theory
class and the nuisance parameters that cannot be factored out (such as the
galaxy bias parameters).
"""
from cobaya.theory import Theory
from scipy.interpolate import interp1d
import pyccl as ccl
import numpy as np


class Limber(Theory):
    """ Computes the angular power spectra
    """

    def initialize(self):
        self.cl_meta = None
        self.l_sample = None
        self.tracer_qs = None
        self.bin_properties = None
        self.is_PT_bias = None
        self.ia_model = None
        self.sample_cen = None
        self.sample_bpw = None
        self.pk_options = None
        self.provider = None

    def initialize_with_provider(self, provider):
        self.provider = provider

    def get_requirements(self):
        return {"CCL": {"methods": {"pk_data": self._get_pk_data}}}

    def must_provide(self, **requirements):
        if "Limber" not in requirements:
            return {}

        options = requirements.get('Limber') or {}
        self.cl_meta = options.get("cl_meta")
        self.l_sample = options.get("l_sample")
        self.tracer_qs = options.get("tracer_qs")
        self.bin_properties = options.get("bin_properties")
        self.is_PT_bias = options.get("is_PT_bias")
        self.ia_model = options.get("ia_model")
        self.sample_cen = options.get("sample_cen")
        self.sample_bpw = options.get("sample_bpw")
        self.pk_options = options.get("pk_options")

        return {}

    def calculate(self, state, want_derived=True, **params_values_dict):
        cosmo = self.provider.get_CCL()["cosmo"]
        state["Limber"] = {"cl_data": self._get_cl_data(cosmo)}

    def get_Limber(self):
        """Get dictionary of Limber computed quantities.
        results['Limber'] contains the computed angular power spectra without
        the parameters that can be factored out.
        Other entries are computed by methods passed in as the requirements

        :return: dict of results
        """
        return self._current_state['Limber']

    def _eval_interp_cl(self, cl_in, l_bpw, w_bpw):
        """ Interpolates C_ell, evaluates it at bandpower window
        ell values and convolves with window."""
        f = interp1d(np.log(1E-3+self.l_sample), cl_in)
        cl_unbinned = f(np.log(1E-3+l_bpw))
        cl_binned = np.dot(w_bpw, cl_unbinned)
        return cl_binned

    def _get_tracers(self, cosmo):
        """ Obtains CCL tracers (and perturbation theory tracers,
        and halo profiles where needed) for all used tracers given the
        current parameters."""
        trs0 = {}
        trs1 = {}
        trs1_dnames = {}
        for name, q in self.tracer_qs.items():
            if q in ["galaxy_density", "galaxy_shear"]:
                z = self.bin_properties[name]['z_fid']
                nz = self.bin_properties[name]['nz_fid']
                oz = np.ones_like(z)

            if q == 'galaxy_density':
                t0 = None
                tr = ccl.NumberCountsTracer(cosmo, dndz=(z, nz),
                                            bias=(z, oz), has_rsd=False)
                t1 = [tr]
                t1n = ['d1']
                if self.is_PT_bias:
                    for bn, dn in zip(['b2', 'bs', 'bk2'], ['d2', 's2', 'k2']):
                        t1.append(tr)
                        t1n.append(dn)
            elif q == 'galaxy_shear':
                t0 = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
                if self.ia_model == 'IANone':
                    t1 = None
                else:
                    t1 = [ccl.WeakLensingTracer(cosmo, dndz=(z, nz),
                                                has_shear=False, ia_bias=None)]
                    t1n = ['m']
            elif q == 'cmb_convergence':
                # B.H. TODO: pass z_source as parameter to the YAML file
                t0 = ccl.CMBLensingTracer(cosmo, z_source=1100)
                t1 = None
                t1n = None

            trs0[name] = t0
            trs1[name] = t1
            trs1_dnames[name] = t1n
        return trs0, trs1, trs1_dnames

    def _get_cl_data(self, cosmo):
        """ Compute all C_ells."""
        # Get P(k)s
        pkd = self.provider.get_CCL()["pk_data"]

        # Gather all tracers
        trs0, trs1, dnames = self._get_tracers(cosmo)

        # Correlate all needed pairs of tracers
        cls_00 = []
        cls_01 = []
        cls_10 = []
        cls_11 = []
        for clm in self.cl_meta:
            if self.sample_cen:
                ls = clm['l_eff']
            elif self.sample_bpw:
                ls = self.l_sample

            n1 = clm['bin_1']
            n2 = clm['bin_2']
            t0_1 = trs0[n1]
            t0_2 = trs0[n2]
            t1_1 = trs1[n1]
            t1_2 = trs1[n2]
            dn_1 = dnames[n1]
            dn_2 = dnames[n2]
            # 00: unbiased x unbiased
            if t0_1 and t0_2:
                pk = pkd['pk_mm']
                cl00 = ccl.angular_cl(cosmo, t0_1, t0_2, ls, p_of_k_a=pk) * clm['pixbeam']
                cls_00.append(cl00)
            else:
                cls_00.append(None)
            # 01: unbiased x biased
            if t0_1 and (t1_2 is not None):
                cl01 = []
                for t12, dn in zip(t1_2, dn_2):
                    pk = pkd[f'pk_m{dn}']
                    if pk is not None:
                        cl = ccl.angular_cl(cosmo, t0_1, t12, ls, p_of_k_a=pk) * clm['pixbeam']
                    else:
                        cl = np.zeros_like(ls)
                    cl01.append(cl)
                cl01 = np.array(cl01)
            else:
                cl01 = None
            cls_01.append(cl01)
            # 10: biased x unbiased
            if n1 == n2:
                cls_10.append(cl01)
            else:
                if t0_2 and (t1_1 is not None):
                    cl10 = []
                    for t11, dn in zip(t1_1, dn_1):
                        pk = pkd[f'pk_m{dn}']
                        if pk is not None:
                            cl = ccl.angular_cl(cosmo, t11, t0_2, ls, p_of_k_a=pk) * clm['pixbeam']
                        else:
                            cl = np.zeros_like(ls)
                        cl10.append(cl)
                    cl10 = np.array(cl10)
                else:
                    cl10 = None
                cls_10.append(cl10)
            # 11: biased x biased
            if (t1_1 is not None) and (t1_2 is not None):
                cl11 = np.zeros([len(t1_1), len(t1_2), len(ls)])
                autocorr = n1 == n2
                for i1, (t11, dn1) in enumerate(zip(t1_1, dn_1)):
                    for i2, (t12, dn2) in enumerate(zip(t1_2, dn_2)):
                        if autocorr and i2 < i1:
                            cl11[i1, i2] = cl11[i2, i1]
                        else:
                            pk = pkd[f'pk_{dn1}{dn2}']
                            if pk is not None:
                                cl = ccl.angular_cl(cosmo, t11, t12, ls, p_of_k_a=pk) * clm['pixbeam']
                            else:
                                cl = np.zeros_like(ls)
                            cl11[i1, i2, :] = cl
            else:
                cl11 = None
            cls_11.append(cl11)
        # Bandpower window convolution
        if self.sample_cen:
            clbs_00 = cls_00
            clbs_01 = cls_01
            clbs_10 = cls_10
            clbs_11 = cls_11
        elif self.sample_bpw:
            clbs_00 = []
            clbs_01 = []
            clbs_10 = []
            clbs_11 = []
            # 00: unbiased x unbiased
            for clm, cl00 in zip(self.cl_meta, cls_00):
                if (cl00 is not None):
                    clb00 = self._eval_interp_cl(cl00, clm['l_bpw'], clm['w_bpw'])
                else:
                    clb00 = None
                clbs_00.append(clb00)
            for clm, cl01, cl10 in zip(self.cl_meta, cls_01, cls_10):
                # 01: unbiased x biased
                if (cl01 is not None):
                    clb01 = []
                    for cl in cl01:
                        clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
                        clb01.append(clb)
                    clb01 = np.array(clb01)
                else:
                    clb01 = None
                clbs_01.append(clb01)
                # 10: biased x unbiased
                if clm['bin_1'] == clm['bin_2']:
                    clbs_10.append(clb01)
                else:
                    if (cl10 is not None):
                        clb10 = []
                        for cl in cl10:
                            clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
                            clb10.append(clb)
                        clb10 = np.array(clb10)
                    else:
                        clb10 = None
                    clbs_10.append(clb10)
                # 11: biased x biased
                for clm, cl11 in zip(self.cl_meta, cls_11):
                    if (cl11 is not None):
                        clb11 = np.zeros((cl11.shape[0], cl11.shape[1], len(clm['l_eff'])))
                        autocorr = clm['bin_1'] == clm['bin_2']
                        for i1 in range(np.shape(cl11)[0]):
                            for i2 in range(np.shape(cl11)[1]):
                                if autocorr and i2 < i1:
                                    clb11[i1, i2] = clb11[i2, i1]
                                else:
                                    cl = cl11[i1,i2,:]
                                    clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
                                    clb11[i1,i2,:] = clb
                    else:
                        clb11 = None
                    clbs_11.append(clb11)

        return {'cl00': clbs_00, 'cl01': clbs_01, 'cl10': clbs_10, 'cl11': clbs_11}

    def _get_pk_data(self, cosmo):
        # TODO: I don't like it reading the pk_options dictionary. I think it'd
        # be better if one could pass the options as argument. Kept like this
        # for now because it's less work
        bias_model = self.pk_options["bias_model"]
        is_PT_bias = self.pk_options["is_PT_bias"]

        cosmo.compute_nonlin_power()
        pkmm = cosmo.get_nonlin_power(name='delta_matter:delta_matter')
        if bias_model == 'Linear':
            pkd = {}
            pkd['pk_mm'] = pkmm
            pkd['pk_md1'] = pkmm
            pkd['pk_d1m'] = pkmm
            pkd['pk_d1d1'] = pkmm
        elif is_PT_bias:
            k_SN_suppress = self.pk_options["k_SN_suppress "]
            if k_SN_suppress > 0:
                k_filter = k_SN_suppress
            else:
                k_filter = None
            if bias_model == 'EulerianPT':
                from .ept import EPTCalculator
                EPTkwargs = self.pk_options["EPTkwargs"]
                ptc = EPTCalculator(with_NC=True, with_IA=False,
                                    log10k_min=EPTkwargs["l10k_min_pks"],
                                    log10k_max=EPTkwargs["l10k_max_pks"],
                                    nk_per_decade=EPTkwargs["nk_per_dex_pks"],
                                    a_arr=EPTkwargs["a_s_pks"],
                                    k_filter=k_filter)
            else:
                raise NotImplementedError("Not yet: " + bias_model)
            pk_lin_z0 = ccl.linear_matter_power(cosmo, ptc.ks, 1.)
            Dz = ccl.growth_factor(cosmo, ptc.a_s)
            ptc.update_pk(pk_lin_z0, Dz)
            pkd = {}
            pkd['pk_mm'] = pkmm
            pkd['pk_md1'] = pkmm
            pkd['pk_md2'] = ptc.get_pk('d1d2')
            pkd['pk_ms2'] = ptc.get_pk('d1s2')
            pkd['pk_mk2'] = ptc.get_pk('d1k2', pgrad=pkmm, cosmo=cosmo)
            pkd['pk_d1m'] = pkd['pk_md1']
            pkd['pk_d1d1'] = pkmm
            pkd['pk_d1d2'] = pkd['pk_md2']
            pkd['pk_d1s2'] = pkd['pk_ms2']
            pkd['pk_d1k2'] = pkd['pk_mk2']
            pkd['pk_d2m'] = pkd['pk_md2']
            pkd['pk_d2d1'] = pkd['pk_d1d2']
            pkd['pk_d2d2'] = ptc.get_pk('d2d2')
            pkd['pk_d2s2'] = ptc.get_pk('d2s2')
            pkd['pk_d2k2'] = None
            pkd['pk_s2m'] = pkd['pk_ms2']
            pkd['pk_s2d1'] = pkd['pk_d1s2']
            pkd['pk_s2d2'] = pkd['pk_d2s2']
            pkd['pk_s2s2'] = ptc.get_pk('s2s2')
            pkd['pk_s2k2'] = None
            pkd['pk_k2m'] = pkd['pk_mk2']
            pkd['pk_k2d1'] = pkd['pk_d1k2']
            pkd['pk_k2d2'] = pkd['pk_d2k2']
            pkd['pk_k2s2'] = pkd['pk_s2k2']
            pkd['pk_k2k2'] = None
        return pkd
