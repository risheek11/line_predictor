'''
This model is based on assumption that our network learns 2 level 2 filters
    -> vertical edges
    -> horizontal edges

    higher level
    -> vertical text group
    -> horizontal text group
'''
import cv2
import os
import pickle
import torch
from torch.optim import SGD, Adam
from torch.optim.rmsprop import RMSprop
from torch import device, cuda
from models.cnn_minimal_architecture import Model
from torch.nn import Module, Sequential, Linear, Sigmoid,SmoothL1Loss, MSELoss
from torch.nn.utils import clip_grad_norm
from utilities import read_json_file, create_dir, LOGGER
from utilities.config import N_ANCHOR_BOXES, EPOCHS, NORMALISED_IMAGES_PATH, \
    REG_HIDDEN_LAYERS, CLS_HIDDEN_LAYERS, VGG_CHANNELS, VGG_SCALE_SIZE, \
    BBOX_XYWH_JSON_PATH, EPOCH_SAVE_INTERVAL, LEARNING_RATE, MOMENTUM, SCHEDULER_GAMMA, \
    MODEL_SAVE_PATH, SCHEDULER_STEP, TEST_IMAGES_PATH, TEST_OUTPUT_PATH, MIN_DIM
from utilities.inference_functions import get_diagonal_from_mpwh, non_max_suppress
from torch.optim.lr_scheduler import StepLR


class Minimal(Module):
    def __init__(self):
        super(Minimal, self).__init__()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        fp = open(MIN_DIM, 'rb')
        self.image_size = pickle.load(fp)
        self.minimal_cnn_network = Model()

        self.width_predicting_regressor = Sequential(
            Linear(1 * 131 * 131, REG_HIDDEN_LAYERS),
            Linear(REG_HIDDEN_LAYERS, 1),
        )

        self.height_predicting_regressor = Sequential(
            Linear(1 * 131 * 131, REG_HIDDEN_LAYERS),
            Linear(REG_HIDDEN_LAYERS, 1),
        )

        self.midpoint_predictor = Sequential(
            Linear(2 * 131 * 131, REG_HIDDEN_LAYERS),
            Linear(REG_HIDDEN_LAYERS, 2),
        )
    def forward(self, img, train = True):
        conv = self.minimal_cnn_network.forward(img)
        horizontal_info_assumption = conv[:, 0, :, :]
        vertical_info_assumption = conv[:, 1, :, :]
        #this to be scaled
        horizontal_info_assumption = horizontal_info_assumption.view(-1, 131 * 131)
        vertical_info_assumption = vertical_info_assumption.view(-1, 131 * 131)
        flattened_conv = conv.view(-1, 2 * 131 * 131)

        predicted_width = self.width_predicting_regressor(horizontal_info_assumption)
        predicted_height = self.height_predicting_regressor(vertical_info_assumption)
        predicted_mid_point = self.midpoint_predictor(flattened_conv)
        return predicted_width, predicted_height, predicted_mid_point

    def model_train(self, epoch_offset=0):
        create_dir(MODEL_SAVE_PATH)
        loss_for_regression = MSELoss()
        img_coors_json = read_json_file(BBOX_XYWH_JSON_PATH)

        optimizer = RMSprop(self.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        # optimizer = Adam(self.parameters(), lr=LEARNING_RATE)
#         optimizer = SGD(self.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)


        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            scheduler.step(epoch)
            LOGGER.debug('Epoch: %s, Current Learning Rate: %s', str(epoch + epoch_offset), str(scheduler.get_lr()))
            for image, coors in img_coors_json.items():
                path_of_image = NORMALISED_IMAGES_PATH + image
                path_of_image = path_of_image.replace('%','_')
                img = cv2.imread(path_of_image)
                img = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0)
                img = img.to(self.device)
                predicted_width, predicted_height, predicted_midpoint = self.forward(img)

                #all are scaled
                mp_x = coors[0][0]
                mp_y = coors[0][1]
                mp = torch.cat((torch.tensor([[mp_x]]).to(self.device),
                                torch.tensor([[mp_y]]).to(self.device)), dim=1).float()

                w = coors[0][2]
                h = coors[0][3]
                loss1 = loss_for_regression(predicted_height, torch.tensor([[h]]).float().to(self.device))
                loss2 = loss_for_regression(predicted_width, torch.tensor([[w]]).float().to(self.device))
                loss3 = loss_for_regression(predicted_midpoint, mp.to(self.device))
                loss = loss1 + loss2 + loss3/2
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(self.parameters(), 0.5)
                optimizer.step()
                epoch_loss = epoch_loss + loss.item()

            if epoch % 5 == 0:
                print('epoch: '+str(epoch)+' '+'loss: '+str(epoch_loss))
            if epoch % EPOCH_SAVE_INTERVAL == 0:
                print('saving')
                torch.save(self.state_dict(), MODEL_SAVE_PATH + 'model_epc_'+str(epoch + epoch_offset)+'.pt')
        torch.save(self.state_dict(), MODEL_SAVE_PATH + 'model_epc_' + str(epoch + epoch_offset) + '.pt')

    def model_inference(self):
        create_dir(TEST_OUTPUT_PATH)
        image_list = os.listdir(TEST_IMAGES_PATH)
        LOGGER.info('Inference begun')
        for image in image_list:
            path_of_image = TEST_IMAGES_PATH + '/' + image
            img = cv2.resize(cv2.imread(path_of_image), (self.image_size, self.image_size))
            img = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0)
            img = img.to(self.device)
            predicted_width, predicted_height, predicted_midpoint = self.forward(img)
            mp_x = predicted_midpoint[0][0].detach().cpu().item()
            mp_y = predicted_midpoint[0][1].detach().cpu().item()
            w = predicted_width[0].detach().cpu().item()
            h = predicted_height[0].detach().cpu().item()
            diag = get_diagonal_from_mpwh([mp_x, mp_y, w, h])
            img = cv2.imread(path_of_image)
            img = cv2.resize(img, (self.image_size, self.image_size))
            im_bbox = cv2.rectangle(img, diag[0], diag[1], (255, 0, 0), 2)
            cv2.imwrite(TEST_OUTPUT_PATH + image, im_bbox)





if __name__ == '__main__':
    import torch
    dummy_img = torch.zeros([1, 3, 592, 592])
    conv = Minimal().forward(dummy_img)
    print(conv.shape)
