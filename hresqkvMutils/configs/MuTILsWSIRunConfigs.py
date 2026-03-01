import os
from pathlib import Path
from pandas import read_csv


# 动态设置 BASEPATH 为当前文件的上一级目录
CURRENT_DIR = Path(__file__).resolve().parent
BASEPATH = CURRENT_DIR.parent  # 假设 Project 是上一级目录


class BaseConfigs:

    DEBUG = False # IMPORTANT: debug??

    # COHORT = 'NiceExamplesForMuTILsPreprint'
    # COHORT = 'TCGA_BRCA'
    # COHORT = 'CPSII_40X'
    #COHORT = 'test30122024'
    # COHORT = 'CPS3_40X'
    # COHORT = 'plco_breast'
    COHORT = '14012025' #input里的, output里的文件夹名

    N_SUBSETS = 16
    # N_SUBSETS = 8 if not DEBUG else 1
    # N_SUBSETS = 16 if not DEBUG else 1

    MODELNAME = 'mutils_06012025'


class GrandChallengeConfigs:

    WORKPATH = BASEPATH / 'TILS_CHALLENGE'
    INPUT_PATH = WORKPATH / '1_INPUT' / BaseConfigs.COHORT
    OUTPUT_PATH = WORKPATH / '2_OUTPUT' / BaseConfigs.COHORT
    BASEMODELPATH = WORKPATH / '0_MODELS' / BaseConfigs.MODELNAME

    restrict_to_vta = False

    save_wsi_mask = True
    save_annotations = True
    save_nuclei_meta = False
    save_nuclei_props = False

    grandch = True

    # Uncomment below to run on GC platform which runs on one slide at a time
    # so the slides will overwrite these files
    gcpaths = {
        'roilocs_in': str(INPUT_PATH / "regions-of-interest.json"),
        'cta2vta': str(WORKPATH / '0_MODELS' / 'Calibrations.json'),
        'roilocs_out': str(OUTPUT_PATH / 'regions-of-interest.json'),
        'result_file': str(OUTPUT_PATH / 'results.json'),
        'tilscore_file': str(OUTPUT_PATH / 'til-score.json'),
        'detect_file': str(OUTPUT_PATH / 'detected-lymphocytes.json'),
        'wsi_mask': str(OUTPUT_PATH / 'images' / 'segmented-stroma'),
    }
    # gcpaths = None  # each slide has its own folder (no overwrite)

    topk_rois = 300
    topk_rois_sampling_mode = "weighted"
    topk_salient_rois = 300
    vlres_scorer_kws = {
        'check_tissue': True,
        'tissue_percent': 25,
        'pixel_overlap': 0,
    }


class FullcTMEConfigs:

    INPUT_PATH = BASEPATH / 'input' / BaseConfigs.COHORT
    OUTPUT_PATH = BASEPATH / 'output' / BaseConfigs.COHORT
    BASEMODELPATH = BASEPATH / 'results' / 'mutils' / 'models' / BaseConfigs.MODELNAME

    restrict_to_vta = False  # True只对标注的svs进行，False对所有的进行

    save_wsi_mask = True
    save_annotations = False
    save_nuclei_meta = True
    save_nuclei_props = True

    grandch = False
    gcpaths = None
    topk_rois = 300
    topk_rois_sampling_mode = "stratified"
    topk_salient_rois = 100
    vlres_scorer_kws = None


class NiceExamplesForMuTILsPreprintConfigs:

    INPUT_PATH = BASEPATH / 'input' / BaseConfigs.COHORT
    OUTPUT_PATH = BASEPATH / 'output' / BaseConfigs.COHORT
    BASEMODELPATH = BASEPATH / 'results' / 'mutils' / 'models' / BaseConfigs.MODELNAME

    restrict_to_vta = False

    save_wsi_mask = True
    save_annotations = False
    save_nuclei_meta = False
    save_nuclei_props = False

    grandch = False
    gcpaths = None
    topk_rois = None
    topk_rois_sampling_mode = "stratified"
    topk_salient_rois = None
    vlres_scorer_kws = None


class RunConfigs:

    # IMPORTANT: THIS SWITCHES THE ANALYSIS TYPE
    # cfg = NiceExamplesForMuTILsPreprintConfigs
    # cfg = GrandChallengeConfigs
    cfg = FullcTMEConfigs

    # 确保输出路径存在
    cfg.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # 确保输入路径存在
    if not cfg.INPUT_PATH.is_dir():
        raise FileNotFoundError(f"指定的输入路径不存在: {cfg.INPUT_PATH}")

    # 配置日志记录器。这必须在导入 histolab 模块之前完成
    from utils.MiscRegionUtils import get_configured_logger, load_region_configs

    LOGDIR = cfg.OUTPUT_PATH / 'LOGS'
    LOGDIR.mkdir(parents=True, exist_ok=True)
    LOGGER = get_configured_logger(
        logdir=str(LOGDIR), prefix='MuTILsWSIRunner', tofile=True
    )

    # 模型权重
    MODEL_PATHS = {}
    for f in range(1, 5):
        MODEL_PATHS[f'{BaseConfigs.MODELNAME}-fold{f}'] = str(
            cfg.BASEMODELPATH / f'fold_{f}' / f'{BaseConfigs.MODELNAME}_fold{f}.pt'
        )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Subdividing cohort to run on multiple docker instances

    from MuTILs_Panoptic.utils.GeneralUtils import splitlist

    ALL_SLIDENAMES = os.listdir(str(cfg.INPUT_PATH))

    if BaseConfigs.COHORT.startswith('TCGA') and cfg.restrict_to_vta:
        # RESTRICT TO ROBERTO SALGADO ASSESSED SLIDES
        SLIDENAMES = read_csv(str(
            BASEPATH / 'data' / 'tcga-clinical' / 'PRIVATE_RSalgado_TCGA_TILScores.csv'
        )).iloc[:, 0].to_list()
        SLIDENAMES = [j[:12] for j in SLIDENAMES]
        SLIDENAMES = [j for j in ALL_SLIDENAMES if j[:12] in SLIDENAMES]

    elif BaseConfigs.COHORT.endswith('CPS2') and cfg.restrict_to_vta:
        # RESTRICT TO TED ASSESSED SLIDES (CPS2)
        acs_vta = read_csv(
            str(BASEPATH / 'data' / 'acs-clinical' / 'CPSII_BRCA_FacilityIDs_20210331.csv'),
            index_col=0
        )
        acs_vta.rename(columns={'TILS_STR': 'vta'}, inplace=True)
        acs_vta = acs_vta.loc[:, 'vta'].map(lambda x: float(x) / 100).dropna()
        SLIDENAMES = list(acs_vta.index)
        SLIDENAMES = [j for j in ALL_SLIDENAMES if j.split('_')[0] in SLIDENAMES]

    else:
        SLIDENAMES = ALL_SLIDENAMES
        SLIDENAMES.sort()
        SLIDENAMES = splitlist(
            SLIDENAMES, len(SLIDENAMES) // BaseConfigs.N_SUBSETS
        )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    RUN_KWARGS = dict(

        # paths & slides
        model_configs=load_region_configs(
            str(cfg.BASEMODELPATH / 'region_model_configs.py'), warn=False
        ),
        model_paths=MODEL_PATHS,
        slides_path=str(cfg.INPUT_PATH),
        base_savedir=str(cfg.OUTPUT_PATH),

        # size params
        # roi_side_hres=512 if BaseConfigs.DEBUG else 1024,
        roi_side_hres=1024,
        discard_edge_hres=0,  # keep 0 -> slow + can't get the gap to be exact
        logger=LOGGER,

        # Defined in cfg
        save_wsi_mask=cfg.save_wsi_mask,
        save_annotations=cfg.save_annotations,
        save_nuclei_meta=cfg.save_nuclei_meta,
        save_nuclei_props=cfg.save_nuclei_props,
        grandch=cfg.grandch,
        gcpaths=cfg.gcpaths,
        topk_rois=cfg.topk_rois,
        topk_rois_sampling_mode=cfg.topk_rois_sampling_mode,
        topk_salient_rois=cfg.topk_salient_rois,
        vlres_scorer_kws=cfg.vlres_scorer_kws,

        _debug=BaseConfigs.DEBUG,
    )
