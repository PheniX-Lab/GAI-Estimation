import timm
import torch
from torch import nn

class model_vit1(torch.nn.Module):
    def __init__(self,RS = 0,CS = 0,num_classes=0,pe = False):
        super().__init__();
        self.model = timm.create_model('vit_large_patch16_384',num_classes=num_classes,pretrained=True,in_chans=3)
        self.time = 0
        self.head = nn.Sequential(
            nn.Linear(2049,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

    def forward_features(self, x):
        self.time=x[1].squeeze(-1)
        x = x[0]
        y = x[:,:3,:,:]
        z = x[:,3:,:,:]
        y = self.model(y)
        z = self.model(z)
        yz = torch.cat((y,z),dim=1)
        return yz

    def forward(self, x):
        x = self.forward_features(x)
        self.time = self.time.unsqueeze(1)
        x = torch.cat((x,self.time), dim=1)
        x = self.head(x)
        return x

class model_vit2(torch.nn.Module):
    def __init__(self,RS = 0,CS = 0,num_classes=0,pe = False):
        super().__init__();
        self.model = timm.create_model('vit_large_patch16_384',num_classes=num_classes,pretrained=True,in_chans=6)
        self.time = 0
        self.head = nn.Sequential(
            nn.Linear(1025,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

    def forward_features(self, x):
        self.time=x[1].squeeze(-1)
        x = self.model(x[0])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        self.time = self.time.unsqueeze(1)
        x = torch.cat((x,self.time), dim=1)
        x = self.head(x)
        return x

class model_vit3(torch.nn.Module):
    def __init__(self,RS = 0,CS = 0,num_classes=0,pe = False):
        super().__init__();
        self.model = timm.create_model('vit_large_patch16_384',num_classes=num_classes,pretrained=True,in_chans=3)
        self.embed_dim = self.model.embed_dim
        self.patch_embed = self.model.patch_embed
        self.cls_token = nn.Parameter(torch.zeros(1,1,self.embed_dim))
        self.scale_embed = nn.Sequential(
            nn.Linear(1,self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed_A = nn.Parameter(torch.zeros(1,num_patches,self.embed_dim))
        self.pos_embed_B = nn.Parameter(torch.zeros(1,num_patches,self.embed_dim))
        self.pos_embed_scale = nn.Parameter(torch.zeros(1,1,self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1,1,self.embed_dim))
        with torch.no_grad():
            self.cls_token.copy_(self.model.cls_token)
            self.pos_embed_cls.copy_(self.model.cls_token)
            self.pos_embed_A.copy_(self.model.pos_embed[:, 1:, :])
            self.pos_embed_B.copy_(self.model.pos_embed[:, 1:, :])
        num_features = 1024
        self.model.head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,1)
        )

    def forward(self,inputs):
        image = inputs[0]
        image_A = image[:,3:,:,:]
        image_B = image[:,:3,:,:]
        B = image_A.shape[0]
        token_A = self.model.patch_embed(image_A)
        token_B = self.model.patch_embed(image_B)
        scale_token = self.scale_embed(inputs[1].float()).unsqueeze(1)
        token_A += self.pos_embed_A
        token_B += self.pos_embed_B
        scale_token += self.pos_embed_scale
        cls_token = self.cls_token.expand(B,-1,-1) + self.pos_embed_cls
        x = torch.cat((cls_token,scale_token,token_A,token_B),dim=1)

        x = self.model.pos_drop(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        cls_output = x[:,0]
        output = self.model.head(cls_output)
        return output
