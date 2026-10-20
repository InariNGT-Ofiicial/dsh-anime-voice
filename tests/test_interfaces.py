"""CPU regressions for previously unsafe training/pipeline interfaces."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import csv
import unittest
import wave

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'skills/anime-voice-clone-training/scripts'


def load(name):
    spec=importlib.util.spec_from_file_location(name,S/(name+'.py'))
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TrainingInterfaces(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'work/test-tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.root=self.base/'gsv'; self.root.mkdir()
        self.ts=self.base/'dataset'; self.ts.mkdir()
        self.audio=[]
        for folder in ['first','second']:
            p=self.base/folder/'same.wav'; p.parent.mkdir()
            with wave.open(str(p),'wb') as w:
                w.setparams((1,2,16000,0,'NONE','not compressed')); w.writeframes(b'\0\0'*1600)
            self.audio.append(p)
        (self.ts/'train.list').write_text('\n'.join(f'{p}|role|ja|test。' for p in self.audio)+'\n',encoding='utf8')

    def run_training(self,*args):
        env={k:v for k,v in os.environ.items() if k not in ['GSV_EXP','GSV_TRAINSET','GSV_ROOT']}
        return subprocess.run([sys.executable,str(S/'train_sovits.py'),'--root',str(self.root),*args],capture_output=True,text=True,env=env)

    def test_missing_exp_and_dataset_fail(self):
        r=self.run_training('--stage','list')
        self.assertNotEqual(r.returncode,0)
        self.assertFalse((self.root/'TEMP').exists())

    def test_list_is_cpu_only_and_collision_safe(self):
        r=self.run_training('--stage','list','--exp','test','--trainset',str(self.ts))
        self.assertEqual(r.returncode,0,r.stderr)
        rows=(self.ts/'gw_list.txt').read_text(encoding='utf8').splitlines()
        paths=[Path(row.split('|')[0]) for row in rows]
        self.assertEqual(len(set(p.name for p in paths)),2)
        self.assertTrue(all(p.is_file() for p in paths))
        self.assertFalse((self.root/'logs').exists(),'list must never start preprocessing')
        self.assertFalse((self.root/'TEMP').exists())

    def test_changed_data_rejects_feature_resume(self):
        m=load('train_sovits'); m.TRAINSET=str(self.ts); m.EXP_DIR=str(self.root/'logs/test')
        m.verify_experiment(create=True)
        p=self.ts/'train.list'; p.write_text(p.read_text(encoding='utf8').replace('test。','changed。'),encoding='utf8')
        with self.assertRaisesRegex(SystemExit,'Dataset changed'): m.verify_experiment()

    def test_existing_unfingerprinted_experiment_refused(self):
        m=load('train_sovits'); m.TRAINSET=str(self.ts); m.EXP_DIR=str(self.root/'logs/test')
        Path(m.EXP_DIR).mkdir(parents=True); (Path(m.EXP_DIR)/'old-feature').touch()
        with self.assertRaisesRegex(SystemExit,'no dataset manifest'): m.verify_experiment(create=True)

    def test_pipeline_train_is_real_driver_call(self):
        r=subprocess.run([sys.executable,str(S/'run_pipeline.py'),'--work',str(self.base),'--stages','train','--exp','test','--trainset',str(self.ts),'--train-stage','list'],capture_output=True,text=True,env={**os.environ,'GSV_ROOT':str(self.root)})
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertTrue((self.ts/'gw_list.txt').is_file())
        status=json.loads((self.base/'pipeline-status.json').read_text(encoding='utf8'))
        self.assertEqual(status[0]['state'],'passed')
        self.assertIn('train_sovits.py',status[0]['argv'][1])

    def test_keep_requires_explicit_valid_spans(self):
        m=load('run_pipeline')
        self.assertEqual(m.parse_keep([]),{})
        for v in ['ep01=10-1','ep01=0-10,9-20']:
            with self.assertRaises(ValueError): m.parse_keep([v])

    def test_export_requires_single_speaker_and_explicit_language(self):
        script=ROOT/'skills/anime-voice-line-extractor/scripts/export_trainset.py'
        inp=self.base/'extract'; inp.mkdir()
        rows=[dict(index=i+1,speaker=spk,start=0,end=1,dur=1,text='test text',asr_conf=1,file=str(self.audio[i])) for i,spk in enumerate(['one','two'])]
        with (inp/'lines.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        r=subprocess.run([sys.executable,str(script),str(inp),'--out',str(self.base/'export')],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('one reviewed target speaker',r.stderr)

    def test_inference_restores_upstream_selection_on_error(self):
        m=load('infer_tts')
        weight=self.root/'weight.json'; weight.write_bytes(b'{"original":true}')
        with self.assertRaises(RuntimeError):
            with m.preserve_weight_selection(self.root):
                weight.write_bytes(b'{"changed":true}')
                raise RuntimeError('inference failed')
        self.assertEqual(weight.read_bytes(),b'{"original":true}')

    def test_inference_removes_only_its_new_selection_file(self):
        m=load('infer_tts')
        weight=self.root/'weight.json'
        with m.preserve_weight_selection(self.root): weight.write_text('{}',encoding='utf8')
        self.assertFalse(weight.exists())


if __name__=='__main__': unittest.main()
