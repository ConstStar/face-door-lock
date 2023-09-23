#include <string.h>

#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "Serial.h"
#include "PWM.h"


#define DOOR_OPEN	5
#define DOOR_CLOSE	25


// ������Կ ��pythonʶ��ϵͳ��� ���ܳ���20λ
#define CONFIG_UNLOCK_KEY "123"


//�����жϴ������
uint8_t KeyTries = 0;

//�����Կ�Ƿ���ȷ
int CheckKey(const uint8_t *RxData,int DataCount)
{
	int i = 0,j = 0;
	int KeyLen = strlen(CONFIG_UNLOCK_KEY);	
	
	if(DataCount < KeyLen)
		return 0;
	
	//ֻ������һ���ַ��Ƿ�����Կ��ͬ
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
	int KeyLen = strlen(CONFIG_UNLOCK_KEY);			// ��¼һ����Կ����
	const uint8_t* RxData = Serial_GetRxData();		// ���մ��ڷ��͹������ַ�����������ַ�����ָ������ֻ��Ҫ����һ��
	int RxCount = 0;
	
	PWM_Init();
	Serial_Init();
	
	while (1)
	{
		RxCount = Serial_GetRxCount();
		if (RxCount >= KeyLen)
		{
			// �����������ﵽ3�� ��3���ڲ������κδ�����Ϣ����ֹ�����ƽ⣩
			if(KeyTries >= 3)
			{
				Delay_s(3);							// 3���ڲ������κ�����
				KeyTries = 0;						// ��մ������
				Serial_ClearRxData();				// ��մ��ڽ��ܵ�����
			}
			else
			{
				// �ж���Կ�Ƿ���ȷ
				if(CheckKey(RxData,RxCount))
				{
					KeyTries = 0;
					Serial_ClearRxData();			// ��մ��ڽ��յ�����
					// ����
					PWM_SetCompare2(DOOR_OPEN);
					Delay_s(5);						// 5����Զ�����
					PWM_SetCompare2(DOOR_CLOSE);
				}
				else
				{
					// ��¼�������
					KeyTries++;
				}
					
			}
		}
	}
}
