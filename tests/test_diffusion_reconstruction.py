import torch
from miniworld.denoiser import DiffusionForcingDenoiser


def test_oracle_flow_velocity_recovers_clean_latent_at_every_noise_level(monkeypatch):
    clean=torch.tensor([1.,2.,3.,4.]).view(1,1,4,1,1)
    noise=torch.tensor([-2.,-1.,0.,1.]).view_as(clean)
    times=torch.tensor([[.05,.25,.75,1.]])
    mask=torch.tensor([[1.,0.,0.,0.]])
    class Oracle(torch.nn.Module):
        def forward(self,*args,**kwargs):return clean-noise
    model=object.__new__(DiffusionForcingDenoiser)
    torch.nn.Module.__init__(model)
    model.net=Oracle()
    model.df_chunk_size=4
    model.drop_cond=lambda x:x
    model._build_diffusion_forcing_timesteps=lambda **kwargs:(times,None,None,mask)
    monkeypatch.setattr(torch,'randn_like',lambda x:noise)
    loss,prediction,peak=model(clean,torch.zeros(1,4,1),return_pred=True)
    torch.testing.assert_close(loss,torch.tensor(0.))
    torch.testing.assert_close(prediction,clean)
    torch.testing.assert_close(peak,torch.ones(1))
