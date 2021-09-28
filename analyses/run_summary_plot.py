from pipeline.pipeline import main_pipeline


conf_files = [
              #'configs/summary_plots_including_phenotype_including_subtypes.yaml',
              #'configs/summary_plots_only_neurobio.yaml',
              'configs/summary_plots_including_phenotype.yaml',
              #'configs/summary_plots_freesurfer.yaml',
              #'configs/summary_plots_hariri.yaml',
              #'configs/summary_plots_connectome_dti_FA.yaml',
              #'configs/summary_plots_connectome_dti_MD.yaml',
              #'configs/summary_plots_connectome_rs.yaml',
              #'configs/summary_plots_graph_metrics_dti.yaml',
              #'configs/summary_plots_graph_metrics_rs.yaml',
              #'configs/summary_plots_prs.yaml',
              #'configs/summary_plots_social_support.yaml',
              #'configs/summary_plots_ctq.yaml',
              ]

for conf_file in conf_files:
    main_pipeline(conf_file)
