from pipeline.pipeline import main_pipeline


debug = True

analyses = {'configs/freesurfer.yaml': False,
            'configs/prs.yaml': False,
            'configs/graph_metrics_dti.yaml': False,
            'configs/graph_metrics_rs.yaml': False,
            'configs/connectome_dti_MD.yaml': False,
            'configs/connectome_dti_FA.yaml': False,
            'configs/connectome_rs.yaml': False,
            'configs/hariri.yaml': False,

            'configs/ctq.yaml': False,
            'configs/social_support.yaml': False,

            'configs/freesurfer_gender.yaml': False,
            'configs/hariri_gender.yaml': True,
            'configs/prs_gender.yaml': False,
            'configs/graph_metrics_dti_gender.yaml': False,
            'configs/graph_metrics_rs_gender.yaml': False,
            'configs/connectome_dti_MD_gender.yaml': False,
            'configs/connectome_dti_FA_gender.yaml': False,
            'configs/connectome_rs_gender.yaml': False,

            'configs/ctq_gender.yaml': False,
            'configs/neuroticism_gender.yaml': False,
            'configs/social_support_gender.yaml': False,
            'configs/ctq_no_bdi_gender.yaml': False,
            'configs/social_support_no_bdi_gender.yaml': False,

            'configs/freesurfer_site.yaml': False,
            'configs/hariri_site.yaml': True,
            'configs/prs_site.yaml': False,
            'configs/graph_metrics_dti_site.yaml': False,
            'configs/graph_metrics_rs_site.yaml': False,
            'configs/connectome_dti_MD_site.yaml': False,
            'configs/connectome_dti_FA_site.yaml': False,
            'configs/connectome_rs_site.yaml': False,

            'configs/ctq_site.yaml': False,
            'configs/social_support_site.yaml': False,

            }

#analyses = {'configs/freesurfer.yaml': True}

task_list = list()
for conf_file, flag in analyses.items():
    if flag:
        if debug:
            main_pipeline(conf_file)
        else:
            try:
                main_pipeline(conf_file)
            except BaseException as e:
                print(e)
