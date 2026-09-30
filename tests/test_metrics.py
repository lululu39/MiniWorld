import json
import math
import sys
import pytest
import torch

from miniworld.metrics import VideoMetrics, psnr_per_frame, rgb_context_frames, summarize_scores, write_report


@pytest.fixture(scope='module')
def evaluator():
    return VideoMetrics('cpu', frame_batch_size=2)


def test_psnr_matches_lvsm_per_view_average():
    gt = torch.zeros(2, 3, 32, 32)
    pred = torch.stack((torch.full_like(gt[0], .1), torch.full_like(gt[0], .2)))
    values = psnr_per_frame(pred, gt)
    torch.testing.assert_close(values, torch.tensor([20., -10*math.log10(.04)]))
    assert abs(values.mean().item() - (-10*math.log10(.025))) > .9
    assert torch.isinf(psnr_per_frame(gt, gt)).all()


def test_three_metrics_identity_context_and_chunks(evaluator):
    torch.manual_seed(9)
    gt = torch.rand(5, 3, 32, 32)*2-1
    pred = gt.clone(); pred[:2] = -gt[:2]
    row = evaluator.score(pred, gt, context_frames=2)
    assert row['mean'] == dict(psnr=float('inf'), ssim=1., lpips=0.)
    assert row['frame_indices'] == [2, 3, 4]
    pred[2:] = (gt[2:] + .2).clamp(-1, 1)
    chunked = evaluator.score(pred, gt, context_frames=2)
    evaluator.frame_batch_size = 8
    batched = evaluator.score(pred, gt, context_frames=2)
    evaluator.frame_batch_size = 2
    for name in ('psnr', 'ssim', 'lpips'):
        torch.testing.assert_close(torch.tensor(chunked['per_frame'][name]), torch.tensor(batched['per_frame'][name]))
    assert chunked['mean']['lpips'] > 0 and chunked['mean']['ssim'] < 1
    # Direct official LPIPS uses [-1,1], with no accidental second normalization.
    with torch.no_grad():
        reference = evaluator.lpips(pred[2:], gt[2:]).flatten()
    torch.testing.assert_close(torch.tensor(batched['per_frame']['lpips']), reference)


def test_ssim_constant_image_analytic(evaluator):
    # RGB [0,1] constants .25 and .5: contrast and structure terms equal 1.
    pred, gt = torch.full((1,3,32,32), -.5), torch.zeros(1,3,32,32)
    row = evaluator.score(pred, gt, context_frames=0)
    expected = (2*.25*.5 + .01**2)/(.25**2 + .5**2 + .01**2)
    assert row['mean']['ssim'] == pytest.approx(expected, abs=2e-4)


def test_report_equal_video_weight_and_standard_json(tmp_path):
    def row(values):
        return dict(evaluated_frames=len(values), frame_indices=list(range(1,len(values)+1)),
                    per_frame={name: values for name in ('psnr','ssim','lpips')},
                    mean={name:sum(values)/len(values) for name in ('psnr','ssim','lpips')})
    rows=[row([10.]), row([20.,40.])]
    summary=summarize_scores(rows)
    assert summary['mean']['psnr']==20.  # each video has equal weight
    assert summary['per_frame'][0]['psnr']==15. and summary['per_frame'][1]['num_videos']==1
    write_report(tmp_path, [row([float('inf')])], {}, {})
    text=(tmp_path/'metrics_summary.json').read_text()
    assert 'Infinity' not in text and json.loads(text)['mean']['psnr']=='inf'
    with pytest.raises(ValueError, match='No videos'):summarize_scores([])


def test_invalid_inputs_and_latent_context(evaluator):
    x=torch.zeros(3,3,32,32)
    assert [rgb_context_frames(n) for n in (1,2,4)]==[1,5,13]
    with pytest.raises(ValueError, match='matching'):evaluator.score(x,x[:2])
    with pytest.raises(ValueError, match='at least one'):evaluator.score(x,x,3)
    with pytest.raises(ValueError, match='floating'):evaluator.score(x.byte(),x.byte())
    with pytest.raises(ValueError, match='Non-finite'):evaluator.score(x+float('nan'),x)


def test_saved_video_cli(tmp_path, monkeypatch, evaluator):
    from miniworld import evaluate
    from miniworld.sample import write_video
    clip=tmp_path/'clip.mp4'
    write_video(str(clip),torch.randint(0,256,(3,32,32,3),dtype=torch.uint8),8)
    manifest=tmp_path/'pairs.jsonl'
    manifest.write_text(json.dumps(dict(id='same',prediction='clip.mp4',target='clip.mp4'))+'\n')
    monkeypatch.setattr(evaluate,'VideoMetrics',lambda *a,**k:evaluator)
    monkeypatch.setattr(sys,'argv',['evaluate','--manifest',str(manifest),'--output_dir',str(tmp_path/'results')])
    evaluate.main()
    summary=json.loads((tmp_path/'results/metrics_summary.json').read_text())
    assert summary['num_videos']==1 and summary['evaluated_frames']==2
    assert summary['mean']==dict(psnr='inf',ssim=1.,lpips=0.)
    with pytest.raises(ValueError, match='available'):evaluate.read_video(clip,num_frames=4)


def test_sampling_metrics_integration(tmp_path, monkeypatch, evaluator):
    from miniworld import sample, metrics
    torch.manual_seed(22)
    gt=torch.rand(9,32,32,3)*2-1
    dataset=[dict(videos=gt,sample_id='clip',source_path='ground_truth.mp4',frame_ids=torch.arange(9))]
    class FakeNet(torch.nn.Module):
        def generate_eval_latents_streaming(self, *args, **kwargs):
            pred=gt.permute(3,0,1,2).unsqueeze(0).clone()
            pred[:,:,:5]=-pred[:,:,:5]  # only observed RGB context is incorrect
            return None,pred
    monkeypatch.setattr(sample,'build_dataset',lambda args:dataset)
    monkeypatch.setattr(sample,'load_wan22_vae',lambda args:None)
    monkeypatch.setattr(sample,'StreamingVAEDecoder',lambda vae:None)
    monkeypatch.setattr(sample,'read_checkpoint',lambda path:({},dict(latent_frames=3,wm_model='B')))
    monkeypatch.setattr(sample,'build_denoiser',lambda args:FakeNet())
    monkeypatch.setattr(sample,'load_weights',lambda *args:None)
    monkeypatch.setattr(sample,'vae_encode',lambda *args:torch.zeros(1,4,3,2,2))
    monkeypatch.setattr(sample,'build_cond_seq_for_batch',lambda **kwargs:torch.zeros(1,3,6))
    monkeypatch.setattr(metrics,'VideoMetrics',lambda *args:evaluator)
    monkeypatch.setattr(sys,'argv',['sample','--dataset','re10k','--data_root','unused','--checkpoint','unused',
          '--vae_checkpoint','unused','--sample_dir',str(tmp_path),'--total_len','3','--history_len','2',
          '--metrics','--benchmark_no_save'])
    sample.main()
    summary=json.loads((tmp_path/'metrics_summary.json').read_text())
    row=json.loads((tmp_path/'metrics_per_video.jsonl').read_text())
    assert summary['evaluated_frames']==4 and row['context_frames']==5
    assert row['sample_id']=='clip' and row['seed']==42
    assert summary['mean']==dict(psnr='inf',ssim=1.,lpips=0.)


def test_strict_dataset_does_not_substitute_bad_samples():
    from miniworld.data.re10k import RealEstate10KDataset
    from pathlib import Path
    ds=object.__new__(RealEstate10KDataset)
    ds.files=[Path('bad.mp4'),Path('good.mp4')];ds.strict_loading=True
    ds._decode_video=lambda path:(_ for _ in ()).throw(ValueError('decode failed'))
    with pytest.raises(RuntimeError,match='bad.mp4'):ds[0]
