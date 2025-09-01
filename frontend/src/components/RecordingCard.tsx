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

  // Load available voices (German only)
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

  // Play target text (auto or custom)
  const handlePlayTarget = () => {
    if (!("speechSynthesis" in window)) return;
    const text = mode === "auto" ? targetText : customText;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "de-DE";
    const voice = voices.find((v) => v.name === selectedVoice);
    if (voice) u.voice = voice;
    u.rate = 1;
    u.pitch = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  };

  // Voice options for Select component
  const voiceItems = createListCollection({
    items: voices.map((v) => ({ label: `${v.name} (${v.lang})`, value: v.name })),
  });

  // Handle text change in custom input
  const handleCustomChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setCustomText(val);
    onTargetTextChange(val);
  };

  return (
    <Card.Root p={6} rounded="2xl" shadow="xl">
      <Card.Header mb={4}>
        <Heading size="lg" textAlign="center">🎤 Pronunciation Practice</Heading>
      </Card.Header>

      <Card.Body>
        <VStack align="center" gap={6} w="full">
          <Box textAlign="center" w="full">
            <Badge colorPalette="blue" rounded="full" px={4} py={2} fontSize="sm">
              Target
            </Badge>

            {/* Mode selector: Auto or Custom */}
            <RadioGroup.Root
              mt={4}
              value={mode}
              onValueChange={({ value }) => setMode(value as "auto" | "custom")}
            >
              <HStack gap={6} justify="center">
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

            {/* Auto mode: show text from API */}
            {mode === "auto" && (
              <HStack justify="center" gap={2} mt={4} wrap="wrap">
                <Text fontSize="xl" fontWeight="semibold">
                  {targetText}
                </Text>
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
            )}

            {/* Custom mode: allow user input */}
            {mode === "custom" && (
              <>
                <Textarea
                  mt={4}
                  rows={6}
                  resize="vertical"
                  placeholder="✍️ Enter your own sentence or paragraph..."
                  value={customText}
                  onChange={handleCustomChange}
                  fontSize="lg"
                  p={4}
                  rounded="xl"
                  shadow="sm"
                  borderColor="teal.400"
                  _focus={{
                    borderColor: "teal.600",
                    shadow: "0 0 0 2px rgba(56,178,172,0.6)",
                  }}
                />
                <HStack justify="space-between" mt={2} w="full">
                  <Text fontSize="sm" color="fg.muted">
                    {customText.length.toLocaleString()} characters
                  </Text>
                  <Button
                    size="md"
                    variant="solid"
                    colorPalette="teal"
                    rounded="lg"
                    shadow="md"
                    onClick={handlePlayTarget}
                    disabled={!customText.trim()}
                  >
                    ▶️ Play Custom Text
                  </Button>
                </HStack>
              </>
            )}

            {/* Select German voice */}
            <Select.Root
              collection={voiceItems}
              size="sm"
              mt={6}
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

          {/* Recording button */}
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

            {/* Recording animation bars */}
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

            {/* Recording status text */}
            <Box aria-live="polite" aria-atomic="true" minH="1.25rem">
              <Text color="fg.muted" fontSize="sm">
                {recording ? "🎙️ Recording…" : "Tap mic to start"}
              </Text>
            </Box>
          </VStack>

          <Separator />

          {/* Next Sentence button only in Auto mode */}
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