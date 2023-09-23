#include <string.h>

#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "Serial.h"
#include "PWM.h"


#define DOOR_OPEN	5
#define DOOR_CLOSE	25


// 开锁秘钥 与python识别系统相符 不能超过20位
#define CONFIG_UNLOCK_KEY "123"


//用来判断错误次数
uint8_t KeyTries = 0;

//检查秘钥是否正确
int CheckKey(const uint8_t *RxData,int DataCount)
{
	int i = 0,j = 0;
	int KeyLen = strlen(CONFIG_UNLOCK_KEY);	
	
	if(DataCount < KeyLen)
		return 0;
	
	//只检查最后一段字符是否与秘钥相同
	i = DataCount - KeyLen;
	j = 0;
	
	for(;i<DataCount;++i,++j){
		if(CONFIG_UNLOCK_KEY[j] != RxData[i])
			return 0;
	}
	
	return 1;
}


int main(void)
{	
	int KeyLen = strlen(CONFIG_UNLOCK_KEY);			// 记录一下密钥长度
	const uint8_t* RxData = Serial_GetRxData();		// 接收串口发送过来的字符串，这个是字符串的指针所以只需要接收一次
	int RxCount = 0;
	
	PWM_Init();
	Serial_Init();
	
	while (1)
	{
		RxCount = Serial_GetRxCount();
		if (RxCount >= KeyLen)
		{
			// 如果错误次数达到3次 则3秒内不接收任何串口信息（防止暴力破解）
			if(KeyTries >= 3)
			{
				Delay_s(3);							// 3秒内不接收任何数据
				KeyTries = 0;						// 清空错误次数
				Serial_ClearRxData();				// 清空串口接受的内容
			}
			else
			{
				// 判断秘钥是否正确
				if(CheckKey(RxData,RxCount))
				{
					KeyTries = 0;
					Serial_ClearRxData();			// 清空串口接收的内容
					// 开门
					PWM_SetCompare2(DOOR_OPEN);
					Delay_s(5);						// 5秒后自动关门
					PWM_SetCompare2(DOOR_CLOSE);
				}
				else
				{
					// 记录错误次数
					KeyTries++;
				}
					
			}
		}
	}
}
