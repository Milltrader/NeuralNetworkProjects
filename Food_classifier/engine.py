import torch
import shutil
import os


def train_step(model,
                train_loader,
                loss_fn,
                optimizer,
                device):

    model.train()

    train_loss, train_acc = 0, 0

    for i_batch, (X_batch, y_batch) in enumerate(train_loader):

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred_logits = model(X_batch)
        loss = loss_fn(y_pred_logits, y_batch)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = y_pred_logits.argmax(dim=1)
        train_acc += (y_pred_class == y_batch).sum().item()/len(y_pred_logits)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc

def val_step(model,
             data_loader,
             loss_fn,
             device):
    
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for i_batch, (X_batch, y_batch) in enumerate(data_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_pred_logits = model(X_batch)
            loss = loss_fn(val_pred_logits, y_batch)
            val_loss += loss.item()
            y_pred_class = val_pred_logits.argmax(dim=1)
            val_acc += (y_pred_class == y_batch).sum().item()/len(y_batch)


        val_loss = val_loss/len(data_loader)
        val_acc = val_acc/len(data_loader)

        return val_loss, val_acc
    
def test_step(model, 
              dataloader, 
              loss_fn,
              device):
  model.eval() 


  test_loss, test_acc = 0, 0

  with torch.inference_mode():

      for batch, (X, y) in enumerate(dataloader):

          X, y = X.to(device), y.to(device)
          # 1. Forward pass
          test_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          # 3. Calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc



def clear_logs(log_dir="logs/fruit_and_vegetable_recognition"):
    """
    Deletes all logs in the specified directory.
    
    Args:
        log_dir (str): Path to the log directory.
    """
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"[INFO] Cleared logs in: {log_dir}")
    else:
        print(f"[INFO] No logs found in: {log_dir}")


def save_model(model, target_dir="models", model_name="model.pth"):
    """
    Saves a PyTorch model to a specified directory.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model.
        target_dir (str): Directory where the model should be saved.
        model_name (str): Name of the saved model file.
    """
    os.makedirs(target_dir, exist_ok=True)  # Create directory if it doesn't exist
    model_path = os.path.join(target_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to: {model_path}")
