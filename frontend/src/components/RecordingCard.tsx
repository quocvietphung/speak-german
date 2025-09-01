"use client";

import React, { useState, useEffect } from "react";
import {
  Box,
  VStack,
  HStack,
  Text,
  Button,
  Heading,
  Card,
  Badge,
  VisuallyHidden,
  Separator,
  IconButton,
  Portal,
  Select,
  RadioGroup,
  createListCollection,
  Textarea,
} from "@chakra-ui/react";
import { keyframes } from "@emotion/react";
import { MdMic, MdStop, MdPlayCircle } from "react-icons/md";

interface RecordingCardProps {
  targetText: string;
  recording: boolean;
  onNextSentence: () => void;
  onRecord: () => void;
  onTargetTextChange: (val: string) => void;
}

const pulse = keyframes`
  0% { transform: scale(1); opacity: .7; }
  70% { transform: scale(1.8); opacity: 0; }
  100% { transform: scale(1.8); opacity: 0; }
`;
const bounce1 = keyframes`0%,100%{height:8px} 50%{height:28px}`;
const bounce2 = keyframes`0%,100%{height:16px} 50%{height:32px}`;
const bounce3 = keyframes`0%,100%{height:10px} 50%{height:26px}`;

export default function RecordingCard({
  targetText,
  recording,
  onNextSentence,
  onRecord,
  onTargetTextChange,
}: RecordingCardProps) {
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [mode, setMode] = useState<"auto" | "custom">("auto");
  const [customText, setCustomText] = useState<string>("");

  useEffect(() => {
    const loadVoices = () => {
      const v = window.speechSynthesis
        .getVoices()
        .filter((x) => x.lang?.toLowerCase().startsWith("de"));
      setVoices(v);
      if (v.length && !selectedVoice) setSelectedVoice(v[0].name);
    };
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }, [selectedVoice]);

  const handlePlayTarget = () => {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(targetText);
    u.lang = "de-DE";
    const voice = voices.find((v) => v.name === selectedVoice);
    if (voice) u.voice = voice;
    u.rate = 1;
    u.pitch = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  };

  const voiceItems = createListCollection({
    items: voices.map((v) => ({ label: `${v.name} (${v.lang})`, value: v.name })),
  });

  const handleCustomChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setCustomText(val);
    onTargetTextChange(val);
  };

  return (
    <Card.Root p={6} rounded="2xl" shadow="xl">
      <Card.Header mb={4}>
        <Heading size="md">🎤 Pronunciation Practice</Heading>
      </Card.Header>

      <Card.Body>
        <VStack align="center" gap={6} w="full">
          <Box textAlign="center" w="full">
            <Badge colorPalette="blue" rounded="full" px={3} py={1}>
              Target
            </Badge>

            <RadioGroup.Root
              mt={4}
              value={mode}
              onValueChange={({ value }) => setMode(value as "auto" | "custom")}
            >
              <HStack gap={4} justify="center">
                <RadioGroup.Item value="auto">
                  <RadioGroup.ItemHiddenInput />
                  <RadioGroup.ItemIndicator />
                  <RadioGroup.ItemText>Auto (Prompt)</RadioGroup.ItemText>
                </RadioGroup.Item>
                <RadioGroup.Item value="custom">
                  <RadioGroup.ItemHiddenInput />
                  <RadioGroup.ItemIndicator />
                  <RadioGroup.ItemText>Custom Input</RadioGroup.ItemText>
                </RadioGroup.Item>
              </HStack>
            </RadioGroup.Root>

            {mode === "auto" ? (
              <>
                <HStack justify="center" gap={2} mt={3} wrap="wrap">
                  <Text fontSize="xl" fontWeight="semibold">{targetText}</Text>
                  <IconButton
                    aria-label="Play target audio"
                    rounded="full"
                    size="sm"
                    colorPalette="teal"
                    variant="subtle"
                    onClick={handlePlayTarget}
                  >
                    <MdPlayCircle />
                  </IconButton>
                </HStack>
              </>
            ) : (
              <>
                <Textarea
                  mt={3}
                  rows={5}
                  resize="vertical"
                  placeholder="Gib deinen eigenen Satz oder Absatz ein..."
                  value={customText}
                  onChange={handleCustomChange}
                />
                <HStack justify="space-between" mt={1} w="full">
                  <Text textStyle="xs" color="fg.muted">
                    {customText.length.toLocaleString()} Zeichen
                  </Text>
                    <Button
                        size="sm"
                        variant="subtle"
                        colorPalette="teal"
                        onClick={() => {
                            if (!("speechSynthesis" in window)) return;
                            const u = new SpeechSynthesisUtterance(customText || "");
                            u.lang = "de-DE";
                            const voice = voices.find((v) => v.name === selectedVoice);
                            if (voice) u.voice = voice;
                            u.rate = 1;
                            u.pitch = 1;
                            window.speechSynthesis.cancel();
                            window.speechSynthesis.speak(u);
                        }}
                        disabled={!customText.trim()}
                    >
                        ▶️ Play Custom Text
                    </Button>
                </HStack>
              </>
            )}

            {/* Chọn giọng nói (Select v3) */}
            <Select.Root
              collection={voiceItems}
              size="sm"
              mt={4}
              multiple={false}
              value={selectedVoice ? [selectedVoice] : []}
              onValueChange={({ value }) => setSelectedVoice(value[0] ?? "")}
            >
              <Select.HiddenSelect />
              <Select.Control>
                <Select.Trigger>
                  <Select.ValueText placeholder="Choose German voice" />
                </Select.Trigger>
              </Select.Control>
              <Portal>
                <Select.Positioner>
                  <Select.Content>
                    {voiceItems.items.map((item) => (
                      <Select.Item key={item.value} item={item}>
                        {item.label}
                        <Select.ItemIndicator />
                      </Select.Item>
                    ))}
                  </Select.Content>
                </Select.Positioner>
              </Portal>
            </Select.Root>
          </Box>

          {/* Nút ghi âm */}
          <VStack gap={3}>
            <Box position="relative" w="96px" h="96px">
              {recording && (
                <>
                  <Box position="absolute" inset="0" rounded="full" bg="pink.400" opacity={0.3} animation={`${pulse} 1.8s ease-out infinite`} />
                  <Box position="absolute" inset="0" rounded="full" bg="purple.400" opacity={0.25} animation={`${pulse} 2s 0.5s ease-out infinite`} />
                </>
              )}
              <Button
                onClick={onRecord}
                aria-pressed={recording}
                aria-label={recording ? "Stop recording" : "Start recording"}
                rounded="full"
                w="96px"
                h="96px"
                fontSize="36px"
                bgGradient="to-br"
                gradientFrom={recording ? "red.400" : "blue.400"}
                gradientTo={recording ? "pink.400" : "teal.400"}
                color="white"
                shadow="2xl"
                _hover={{ transform: "scale(1.05)" }}
                _active={{ transform: "scale(0.96)" }}
              >
                {recording ? <MdStop /> : <MdMic />}
              </Button>
              <VisuallyHidden>
                <span>{recording ? "Recording…" : "Start recording"}</span>
              </VisuallyHidden>
            </Box>

            <HStack gap={2} h="30px" align="end">
              {[bounce1, bounce2, bounce3].map((anim, i) => (
                <Box
                  key={i}
                  w="10px"
                  rounded="sm"
                  bg={recording ? "purple.400" : "gray.300"}
                  animation={recording ? `${anim} 1s ease-in-out ${i * 0.1}s infinite` : undefined}
                />
              ))}
            </HStack>

            <Box aria-live="polite" aria-atomic="true" minH="1.25rem">
              <Text color="fg.muted" textStyle="sm">
                {recording ? "🎙️ Recording…" : "Tap mic to start"}
              </Text>
            </Box>
          </VStack>

          <Separator />

          {/* Next Sentence CHỈ hiển thị ở Auto */}
          {mode === "auto" && (
            <Button
              onClick={onNextSentence}
              colorPalette="teal"
              variant="solid"
              size="md"
              w="full"
              rounded="xl"
              shadow="md"
              disabled={recording}
            >
              ➡️ Next Sentence
            </Button>
          )}
        </VStack>
      </Card.Body>
    </Card.Root>
  );
}